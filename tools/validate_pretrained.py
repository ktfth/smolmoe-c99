#!/usr/bin/env python3
"""Execute the bundled trained model and compare exact budget/backend results.
No downloads or ML frameworks. Optional upstream reference is an external CLI.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate(root, directory, report, reference=None, original=None, original_tokenizer=None):
    prompt = "Once upon a time, there was a little robot"
    base = ["--model", str(directory / "model.smoe"), "--config", str(directory / "config.ini"),
            "--tokenizer", str(directory / "tokenizer.spb"), "--prompt", prompt, "--context", "256"]
    observations, baseline = [], None
    with tempfile.TemporaryDirectory(prefix="smolmoe-pretrained-") as temp:
        trace, ids = Path(temp) / "trace", Path(temp) / "ids"
        for budget in (8, 16, 32, 64):
            for scalar in (False, True):
                command = [str(root / "smolmoe")] + base + ["--tokens", "24", "--memory-mib", str(budget),
                                                           "--trace", str(trace), "--token-ids", str(ids)]
                if scalar:
                    command += ["--scalar"]
                p = subprocess.run(command, capture_output=True, check=True, timeout=60)
                s = json.loads(p.stderr)
                observed = (hashlib.sha256(p.stdout).hexdigest(), digest(trace), digest(ids))
                if baseline is None:
                    baseline = observed
                if observed != baseline:
                    raise AssertionError(f"budget/backend output mismatch: {budget} MiB, scalar={scalar}")
                if s["tracked_peak_bytes"] > budget * 1048576:
                    raise AssertionError("allocation budget exceeded")
                observations.append(dict(budget_mib=budget, backend=s["backend"],
                                         output_sha256=observed[0], logits_sha256=observed[1], ids_sha256=observed[2],
                                         tracked_peak_bytes=s["tracked_peak_bytes"], rss_peak_bytes=s["rss_peak_bytes"],
                                         read_bytes=s["read_bytes"]))
        # A separate untraced run measures generation without trace formatting.
        p = subprocess.run([str(root / "smolmoe")] + base + ["--tokens", "80", "--memory-mib", "16"],
                           capture_output=True, check=True, timeout=60)
        stats = json.loads(p.stderr)
        (directory / "output.txt").write_bytes(p.stdout)
        (directory / "run.json").write_text(json.dumps(stats, indent=2) + "\n")
        # Second prompt verifies a real change in continuation.
        b2 = list(base)
        b2[b2.index("--prompt") + 1] = "The little bird found a"
        second = subprocess.run([str(root / "smolmoe")] + b2 + ["--tokens", "48", "--memory-mib", "8"],
                                capture_output=True, check=True, timeout=60)
        if second.stdout == p.stdout:
            raise AssertionError("different prompts produced the same full continuation")
        result = dict(model="karpathy/tinyllamas stories15M, converted Q8 (lossy)",
                      prompt=prompt, continuation=p.stdout.decode("utf-8"), stats_without_trace=stats,
                      exact_budget_backend_checks=observations,
                      second_prompt=b2[b2.index("--prompt") + 1], second_continuation=second.stdout.decode("utf-8"),
                      original_f32_reference_checked=False,
                      notes="Single host; warm OS filesystem cache possible. Not a Kimi K3 or general chat evaluation.")
        if reference:
            if not original or not original_tokenizer:
                raise ValueError("reference check requires original model and tokenizer")
            steps = stats["prompt_tokens"] + stats["generated_tokens"] - 1
            ref = subprocess.run([str(reference), str(original), "-z", str(original_tokenizer),
                                  "-i", prompt, "-t", "0", "-n", str(steps)],
                                 capture_output=True, check=True, timeout=60)
            expected = prompt.encode() + p.stdout + b"\n"
            if ref.stdout != expected:
                raise AssertionError("greedy reference continuation differs (Q8 is lossy; inspect the mismatch)")
            result.update(original_f32_reference_checked=True,
                          original_f32_greedy_output_equal_for_this_prompt=True,
                          original_f32_sha256=digest(original), reference_binary_sha256=digest(reference))
    Path(report).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": "passed", "budget_backend_comparisons": len(observations),
                      "prompt": prompt, "continuation": result["continuation"],
                      "stats": stats, "reference_checked": result["original_f32_reference_checked"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, default=Path("docs/pretrained-validation.json"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--original-tokenizer", type=Path)
    args = parser.parse_args()
    validate(Path(__file__).resolve().parents[1], args.directory.resolve(), args.report,
             args.reference, args.original, args.original_tokenizer)
