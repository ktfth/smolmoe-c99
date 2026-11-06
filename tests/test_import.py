"""Offline regression checks for trained-checkpoint import and score BPE."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from test_runtime import ROOT
sys.path.insert(0, str(ROOT / "tools"))
import checkpoint as ck
import import_llama2c as imp
import reference


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.source = self.work / "legacy.bin"
        self.tokenizer = self.work / "tokenizer.bin"
        self.output = self.work / "converted"
        # Genuine legacy layout, synthetic values for offline numerical tests.
        d, h, layers, heads, kv_heads, vocab, ctx = 8, 12, 1, 2, 1, 264, 64
        import random
        rng = random.Random(9)
        with self.source.open("wb") as f:
            f.write(struct.pack("<7i", d, h, layers, heads, kv_heads, vocab, ctx))
            shapes = [(vocab, d), (1, d), (d, d), (4, d), (4, d), (d, d),
                      (1, d), (h, d), (d, h), (h, d), (1, d), (ctx, d // heads)]
            for index, (rows, cols) in enumerate(shapes):
                for _ in range(rows):
                    row = [1.0] * cols if index in (1, 6, 10) else [rng.uniform(-0.1, 0.1) for _ in range(cols)]
                    f.write(struct.pack(f"<{cols}f", *row))
        pieces = [(0.0, b"<unk>"), (0.0, b"\n<s>\n"), (0.0, b"\n</s>\n")]
        pieces += [(0.0, f"<0x{b:02X}>".encode()) for b in range(256)]
        pieces += [(-10.0, b" "), (-5.0, b"H"), (-4.0, b"i"), (-3.0, b"Hi"), (-2.0, b" Hi")]
        with self.tokenizer.open("wb") as f:
            f.write(struct.pack("<I", 16))
            for score, piece in pieces:
                f.write(struct.pack("<fI", score, len(piece)) + piece)
        self.c = imp.convert(self.source, self.tokenizer, self.output)

    def tearDown(self):
        self.tmp.cleanup()

    def command(self, extra=(), ok=True):
        p = subprocess.run([str(ROOT / "smolmoe"), "--model", str(self.output / "model.smoe"),
                            "--config", str(self.output / "config.ini"), "--tokenizer", str(self.output / "tokenizer.spb"),
                            "--prompt", "Hi", "--context", "64", "--tokens", "4", "--memory-mib", "2"] + list(map(str, extra)),
                           capture_output=True, timeout=30)
        self.assertEqual(p.returncode == 0, ok, p.stderr.decode(errors="replace"))
        return p

    def test_score_bpe_merges_and_roundtrip(self):
        ids = self.work / "ids"
        p = self.command(("--tokenize-only", "--token-ids", ids))
        self.assertEqual(p.stdout, b"Hi")
        self.assertEqual(ids.read_text(), "256\n263\n")  # original ID 263 survives permutation
        prompt = "Hi Olá 🌳\n".encode() + b"\x00"
        path = self.work / "prompt.bin"
        path.write_bytes(prompt)
        args = [str(ROOT / "smolmoe"), "--config", str(self.output / "config.ini"),
                "--tokenizer", str(self.output / "tokenizer.spb"), "--prompt-file", str(path), "--tokenize-only"]
        p = subprocess.run(args, capture_output=True, check=True)
        self.assertEqual(p.stdout, prompt)

    def test_imported_logits_independent_reference(self):
        trace = self.work / "trace"
        self.command(("--tokens", 0, "--trace", trace))
        actual = [[float.fromhex(s) for s in row.split()[1:]] for row in trace.read_text().splitlines()]
        expected = reference.run(self.c, reference.load(self.output / "model.smoe"), [256, 263])
        for a, b in zip(actual, expected):
            self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 2e-6)

    def test_import_shape_and_token_remapping(self):
        w = reference.load(self.output / "model.smoe")
        self.assertEqual(w["embed"], w["lm_head"])
        self.assertEqual(w["l.0.router"], [[0.0] * 8])
        self.assertEqual((imp.old_id(0), imp.old_id(255), imp.old_id(256), imp.old_id(257), imp.old_id(258)), (3, 258, 1, 2, 0))
        record = json.loads((self.output / "provenance.json").read_text())
        self.assertEqual(record["model_sha256"], imp.sha256(self.source))

    def test_corrupt_score_bpe_rejected(self):
        path = self.output / "tokenizer.spb"
        raw = path.read_bytes()
        for change in (raw[:-1], raw + b"x", raw[:12] + struct.pack("<I", 999999) + raw[16:]):
            path.write_bytes(change)
            self.command(ok=False)

    def test_import_source_size_validation(self):
        with self.source.open("ab") as f:
            f.write(b"x")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            imp.convert(self.source, self.tokenizer, self.work / "bad")

    def test_no_special_alias_overwrite_and_tiny_budget(self):
        p = self.command(("--memory-mib", "1"))
        self.assertLessEqual(json.loads(p.stderr)["tracked_peak_bytes"], 1048576)


if __name__ == "__main__":
    unittest.main()
