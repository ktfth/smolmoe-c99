#!/usr/bin/env python3
"""Convert legacy llama2.c F32 files to smolmoe Q8, row by row.

Supported: legacy 7-int header, adjacent-pair RoPE, SiLU, RMSNorm eps=1e-5,
no biases, optional tied embeddings. Dense FFNs are represented as one expert
with a zero router (normalized weight exactly 1). This is NOT Kimi conversion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import checkpoint as ck

MODEL_URL = "https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories15M.bin"
TOKENIZER_URL = "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin"
MODEL_SHA256 = "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
TOKENIZER_SHA256 = "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def old_id(new):
    if new < 256:
        return new + 3
    return {256: 1, 257: 2, 258: 0}.get(new, new)


def tokenizer_convert(source, output, vocab):
    with Path(source).open("rb") as f:
        maximum, = struct.unpack("<I", f.read(4))
        if not 0 < maximum <= 65536:
            raise ValueError("invalid tokenizer max length")
        pieces = []
        for _ in range(vocab):
            score, length = struct.unpack("<fI", f.read(8))
            if not 0 < length <= maximum:
                raise ValueError("invalid tokenizer piece length")
            piece = f.read(length)
            if len(piece) != length:
                raise ValueError("truncated tokenizer")
            pieces.append((score, piece))
        if f.read(1):
            raise ValueError("tokenizer has extra records")
    for byte in range(256):
        if pieces[byte + 3][1] != f"<0x{byte:02X}>".encode():
            raise ValueError("expected llama2.c byte fallback token IDs 3..258")
    if len({p for _, p in pieces}) != vocab:
        raise ValueError("duplicate tokenizer pieces")
    with Path(output).open("wb") as f:
        f.write(b"SMSPB1\0\1" + struct.pack("<III", vocab, maximum, sum(len(p) for _, p in pieces)))
        for new in range(vocab):
            score, piece = pieces[old_id(new)]
            f.write(struct.pack("<fI", score, len(piece)) + piece)


def convert(model, tokenizer, output):
    model, output = Path(model), Path(output)
    with model.open("rb") as f:
        d, h, layers, heads, kv_heads, signed_vocab, context = struct.unpack("<7i", f.read(28))
    vocab = abs(signed_vocab)
    c = dict(ck.BASE_CONFIG, dim=d, hidden_dim=h, layers=layers, heads=heads,
             kv_heads=kv_heads, experts=1, top_k=1, shared_experts=0,
             vocab_size=vocab, max_context=context)
    output.mkdir(parents=True, exist_ok=True)
    ck.write_config(output / "config.ini", c)
    ck.read_config(output / "config.ini")  # validate before iterating dimensions
    kd = kv_heads * (d // heads)
    offset, slices = 28, {}
    def block(name, count, rows, cols):
        nonlocal offset
        slices[name] = (offset, rows, cols)
        offset += count * rows * cols * 4
    block("embed", 1, vocab, d)
    block("attn_norm", layers, 1, d)
    block("q", layers, d, d)
    block("k", layers, kd, d)
    block("v", layers, kd, d)
    block("o", layers, d, d)
    block("ffn_norm", layers, 1, d)
    block("gate", layers, h, d)
    block("down", layers, d, h)
    block("up", layers, h, d)
    block("final_norm", 1, 1, d)
    offset += context * (d // heads) * 4  # legacy real/imag RoPE arrays
    if signed_vocab > 0:
        slices["lm_head"] = slices["embed"]
    else:
        block("lm_head", 1, vocab, d)
    if offset != model.stat().st_size:
        raise ValueError(f"legacy checkpoint size mismatch: expected {offset}, got {model.stat().st_size}")
    def rows(name, nr, nc):
        if name in ("embed", "lm_head", "final_norm"):
            base, _, _ = slices[name]
            layer = 0
        else:
            parts = name.split(".")
            layer = int(parts[1])
            if parts[2] == "router":
                yield [0.0] * nc
                return
            suffix = parts[-1]
            base, sr, sc = slices[suffix]
            base += layer * sr * sc * 4
        with model.open("rb") as f:
            for r in range(nr):
                source_row = old_id(r) if name in ("embed", "lm_head") else r
                f.seek(base + source_row * nc * 4)
                raw = f.read(nc * 4)
                if len(raw) != nc * 4:
                    raise ValueError(f"truncated tensor {name}")
                yield struct.unpack(f"<{nc}f", raw)
    ck.write_checkpoint(output / "model.smoe", ck.shapes(c), rows)
    tokenizer_convert(tokenizer, output / "tokenizer.spb", vocab)
    record = dict(source_format="legacy llama2.c F32", model_sha256=sha256(model),
                  tokenizer_sha256=sha256(tokenizer), quantization="Q8 block32 F32 scale, lossy",
                  source_vocab_size=vocab, token_remapping="byte n -> n+3; BOS256->1; EOS257->2; UNK258->0; others unchanged",
                  dense_mapping="one expert, top_k=1, no shared experts, zero router",
                  full_kimi_compatible=False)
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return c


def download(url, target, expected):
    target = Path(target)
    if target.exists() and sha256(target) == expected:
        return
    temporary = target.with_suffix(".partial")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "smolmoe-checkpoint-import/1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as f:
            for block in iter(lambda: response.read(1 << 20), b""):
                f.write(block)
        if sha256(temporary) != expected:
            raise ValueError("download checksum mismatch")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="legacy F32 checkpoint path")
    parser.add_argument("tokenizer", help="original llama2.c tokenizer.bin path")
    parser.add_argument("output", help="converted model directory")
    parser.add_argument("--fetch-tiny15m", action="store_true", help="download and verify official 15M checkpoint and tokenizer first")
    args = parser.parse_args()
    try:
        if args.fetch_tiny15m:
            for path in (args.model, args.tokenizer):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            download(MODEL_URL, args.model, MODEL_SHA256)
            download(TOKENIZER_URL, args.tokenizer, TOKENIZER_SHA256)
        c = convert(args.model, args.tokenizer, args.output)
        print(f"Converted trained checkpoint: dim={c['dim']}, layers={c['layers']}, vocab={c['vocab_size']}")
    except (OSError, ValueError, struct.error) as e:
        parser.exit(1, f"import: {e}\n")


if __name__ == "__main__":
    main()
