import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import checkpoint as ck  # noqa: E402
import reference  # noqa: E402


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="smolmoe-tests-")
        cls.work = Path(cls.tmp.name)
        cls.small = cls.work / "small"
        cls.config = ck.fixture(cls.small)
        cls.medium = cls.work / "medium"
        c = dict(ck.BASE_CONFIG, dim=64, hidden_dim=96, experts=64)
        ck.fixture(cls.medium, c)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def call(self, directory=None, extra=(), binary="smolmoe", ok=True, preexec_fn=None):
        d = directory or self.small
        command = [str(ROOT / binary), "--model", str(d / "model.smoe"),
                   "--config", str(d / "config.ini"), "--tokenizer", str(d / "tokenizer.bpe"),
                   "--prompt", "Hi!", "--tokens", "8", "--context", "64", "--memory-mib", "4"]
        p = subprocess.run(command + list(map(str, extra)), capture_output=True,
                           timeout=45, preexec_fn=preexec_fn)
        self.assertEqual(p.returncode == 0, ok, p.stderr.decode(errors="replace"))
        if not ok:
            self.assertIn(b"smolmoe:", p.stderr)
        return p

    def test_exact_logits_across_budgets_and_backends(self):
        expected = None
        reads = []
        for budget in (1, 2, 4, 16, 32):
            for binary in ("smolmoe", "smolmoe-scalar"):
                trace = self.work / "trace"
                ids = self.work / "ids"
                p = self.call(self.medium, ("--memory-mib", budget, "--trace", trace, "--token-ids", ids), binary)
                observed = (p.stdout, trace.read_bytes(), ids.read_bytes())
                if expected is None:
                    expected = observed
                self.assertEqual(expected, observed, (budget, binary))
                stats = json.loads(p.stderr)
                self.assertLessEqual(stats["tracked_peak_bytes"], budget * 1048576)
                self.assertEqual(stats["expert_calls"], stats["processed_tokens"] * 2 * 3)
                self.assertGreater(stats["expert_read_bytes"], 0)
                if binary == "smolmoe":
                    reads.append(stats["read_bytes"])
        self.assertGreater(reads[0], reads[-1], reads)

    def test_reference_full_logits(self):
        trace = self.work / "reference.trace"
        ids = self.work / "reference.ids"
        self.call(extra=("--trace", trace, "--token-ids", ids))
        actual = [[float.fromhex(s) for s in line.split()[1:]] for line in trace.read_text().splitlines()]
        token_ids = [int(x) for x in ids.read_text().splitlines()][:len(actual)]
        self.assertEqual(token_ids[:2], [256, 259])  # BOS, merged Hi!
        expected = reference.run(self.config, reference.load(self.small / "model.smoe"), token_ids)
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(len(a), len(e))
            self.assertLess(max(abs(x - y) for x, y in zip(a, e)), 2e-6)

    def test_seeded_sampling_budget_invariance(self):
        outputs = [self.call(self.medium, ("--temperature", 0.8, "--seed", 42,
                                         "--tokens", 24, "--memory-mib", budget)).stdout
                   for budget in (1, 4, 16)]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_prompt_and_tokenizer_roundtrip(self):
        content = b"Hi!\x00Ol\xc3\xa1 \xf0\x9f\x8c\xb3\n" + bytes(range(256))
        path, ids = self.work / "prompt.bin", self.work / "tokens.txt"
        path.write_bytes(content)
        command = [str(ROOT / "smolmoe"), "--config", str(self.small / "config.ini"),
                   "--tokenizer", str(self.small / "tokenizer.bpe"), "--prompt-file", str(path),
                   "--tokenize-only", "--token-ids", str(ids), "--context", "512", "--memory-mib", "1"]
        p = subprocess.run(command, capture_output=True, check=True)
        self.assertEqual(content, p.stdout)
        self.assertEqual(ids.read_text().splitlines()[:2], ["256", "259"])

    def test_empty_prompt_bos(self):
        ids = self.work / "empty.ids"
        p = self.call(extra=("--prompt", "", "--tokens", 0, "--token-ids", ids))
        self.assertEqual(ids.read_text(), "256\n")
        self.assertEqual(json.loads(p.stderr)["prompt_tokens"], 1)

    def test_sparse_offsets_beyond_8gib_under_address_limit(self):
        d = self.work / "sparse"
        ck.fixture(d, gap=9 << 30)
        self.assertGreater((d / "model.smoe").stat().st_size, 8 << 30)
        normal = self.call(extra=("--memory-mib", 1))
        def limit():
            resource.setrlimit(resource.RLIMIT_AS, (64 << 20, 64 << 20))
        # ASan reserves a large shadow address space; ordinary runs use 64 MiB.
        sanitized = b"__asan_init" in (ROOT / "smolmoe").read_bytes()
        p = self.call(d, ("--memory-mib", 1), preexec_fn=None if sanitized else limit)
        self.assertEqual(normal.stdout, p.stdout)
        self.assertLess(json.loads(p.stderr)["read_bytes"], 8 << 20)
        (d / "model.smoe").unlink()  # do not archive a sparse file by accident

    def test_stats_file_and_forced_scalar(self):
        path = self.work / "stats.json"
        p = self.call(extra=("--stats", path, "--scalar", "--drop-cache"))
        stats = json.loads(path.read_text())
        self.assertEqual(stats, json.loads(p.stderr))
        self.assertEqual(stats["backend"], "scalar")
        self.assertGreater(stats["rss_peak_bytes"], 0)
        self.assertGreater(stats["prefill_tokens_per_second"], 0)
        self.assertEqual(stats["kv_bytes"], 2 * 64 * 16 * 8)

    def test_q8_quantizer_error_bound(self):
        row = [(-1) ** i * i / 13 for i in range(43)]
        raw = ck.q8_row(row)
        for block in range(2):
            scale, *q = struct.unpack_from("<f32b", raw, block * 36)
            for x, v in zip(row[block * 32:(block + 1) * 32], q):
                self.assertLessEqual(abs(x - scale * v), scale / 2 + 1e-6)

    def test_actual_checkpoint_larger_than_budget(self):
        size = (self.medium / "model.smoe").stat().st_size
        self.assertGreater(size, 2 * 1048576)
        stats = json.loads(self.call(self.medium, ("--memory-mib", 1)).stderr)
        self.assertLess(stats["tracked_peak_bytes"], 1048576)

    def test_eos_stops_generation(self):
        d = self.work / "eos"
        c = ck.fixture(d, echo=True)
        specs = ck.shapes(c)
        def rows(name, nr, nc):
            for r in range(nr):
                if name.endswith("norm") or name == "embed" or (name == "lm_head" and r == 257):
                    yield [1.0] * nc
                else:
                    yield [0.0] * nc
        ck.write_checkpoint(d / "model.smoe", specs, rows)
        stats = json.loads(self.call(d).stderr)
        self.assertEqual(stats["generated_tokens"], 0)
        self.assertEqual(stats["processed_tokens"], 4)  # no BPE in this fixture

    def test_packer_streaming_roundtrip(self):
        source, out = self.work / "f32-source", self.work / "packed"
        source.mkdir()
        weights = reference.load(self.small / "model.smoe")
        manifest = {}
        for name, matrix in weights.items():
            path = source / f"{name}.f32"
            with path.open("wb") as f:
                for row in matrix:
                    f.write(struct.pack(f"<{len(row)}f", *row))
            manifest[name] = {"file": path.name}
        (source / "manifest.json").write_text(json.dumps(manifest))
        (source / "config.ini").write_bytes((self.small / "config.ini").read_bytes())
        (source / "tokenizer.bpe").write_bytes((self.small / "tokenizer.bpe").read_bytes())
        ck.pack(source, out)
        # Quantization idempotence is bounded by roundoff, verify numerically.
        a, b = reference.load(self.small / "model.smoe"), reference.load(out / "model.smoe")
        for name in a:
            self.assertLess(max(abs(x - y) for r, s in zip(a[name], b[name]) for x, y in zip(r, s)), 1e-7)
        self.call(out)

    def test_invalid_cli_values(self):
        for extra in (("--temperature", "nan"), ("--temperature", -1), ("--seed", 0),
                      ("--memory-mib", 0), ("--context", 513), ("--tokens", 80),
                      ("--memory-mib", "-1"), ("--unknown", "x"), ("--tokens", "3junk")):
            with self.subTest(extra=extra):
                self.call(extra=extra, ok=False)

    def test_output_cannot_overwrite_checkpoint_or_config(self):
        before = (self.small / "model.smoe").read_bytes()
        self.call(extra=("--stats", self.small / "model.smoe"), ok=False)
        self.call(extra=("--trace", self.small / "config.ini"), ok=False)
        link = self.work / "model-link"
        link.symlink_to(self.small / "model.smoe")
        self.call(extra=("--token-ids", link), ok=False)
        self.assertEqual(before, (self.small / "model.smoe").read_bytes())

    def test_budget_rejection_before_large_kv_allocation(self):
        d = self.bad_copy("large-context")
        config = d / "config.ini"
        config.write_text(config.read_text().replace("max_context=512", "max_context=8192"))
        p = self.call(d, ("--context", 8192, "--memory-mib", 1), ok=False)
        self.assertIn(b"memory budget too small", p.stderr)

    def test_selected_tensor_larger_than_entire_cache(self):
        d = self.work / "large-expert"
        c = dict(ck.BASE_CONFIG, dim=64, hidden_dim=8192, layers=1,
                 experts=1, top_k=1, shared_experts=0)
        ck.fixture(d, c)
        a = self.call(d, ("--memory-mib", 1))
        b = self.call(d, ("--memory-mib", 8))
        self.assertEqual(a.stdout, b.stdout)
        stats = json.loads(a.stderr)
        entry = next(e for e in ck.inspect(d / "model.smoe") if e["name"] == "l.0.e.0.gate")
        self.assertGreater(entry["bytes"], stats["cache_bytes"])

    def test_prompt_over_context_rejected(self):
        self.call(extra=("--prompt", "x" * 64), ok=False)

    def bad_copy(self, name):
        d = self.work / name
        d.mkdir()
        for f in self.small.iterdir():
            (d / f.name).write_bytes(f.read_bytes())
        return d

    def test_truncated_checkpoint(self):
        d = self.bad_copy("truncated")
        with (d / "model.smoe").open("r+b") as f:
            f.truncate(100)
        self.call(d, ok=False)

    def test_bad_extent_and_shape(self):
        for what, relative_offset, value in (("extent", 56, (1 << 64) - 1), ("shape", 44, 0)):
            d = self.bad_copy(what)
            with (d / "model.smoe").open("r+b") as f:
                f.seek(16 + relative_offset)
                f.write(struct.pack("<Q" if what == "extent" else "<I", value))
            self.call(d, ok=False)

    def test_nonfinite_scale_rejected(self):
        d = self.bad_copy("nan")
        embed = next(e for e in ck.inspect(d / "model.smoe") if e["name"] == "embed")
        with (d / "model.smoe").open("r+b") as f:
            f.seek(embed["offset"] + 256 * 36)
            f.write(struct.pack("<f", float("nan")))
        self.call(d, ok=False)

    def test_bad_config_and_kimi_explicit_rejection(self):
        for what, change in (("duplicate", "dim=32\n"), ("unknown", "mystery=1\n")):
            d = self.bad_copy(what)
            with (d / "config.ini").open("a") as f:
                f.write(change)
            self.call(d, ok=False)
        d = self.bad_copy("kimi")
        path = d / "config.ini"
        path.write_text(path.read_text().replace("smolmoe_v1", "kimi_k3"))
        self.assertIn(b"full Kimi K3 is not supported", self.call(d, ok=False).stderr)

    def test_invalid_bpe(self):
        for name, text in (("forward-bpe", "SMBPE1\n258 1\n258 33\n"),
                           ("duplicate-bpe", "SMBPE1\n72 105\n72 105\n"),
                           ("special-bpe", "SMBPE1\n256 1\n258 33\n")):
            d = self.bad_copy(name)
            (d / "tokenizer.bpe").write_text(text)
            self.call(d, ok=False)


if __name__ == "__main__":
    unittest.main()
