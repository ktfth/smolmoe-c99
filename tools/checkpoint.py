#!/usr/bin/env python3
"""Dependency-free row-streaming Q8 packer, synthetic fixtures and inspector.

Python is a development/conversion dependency only; inference is entirely C.
Input: canonical tensor names and IEEE little-endian F32 raw tensor files.
No implicit conversion of Kimi, GGUF, MXFP4 or Hugging Face architectures.
"""
import argparse
import configparser
import json
import math
from pathlib import Path
import random
import shutil
import struct
import sys

MAGIC = b"SMOEQ8\0\1"
RECORD = struct.Struct("<40sIIIIQQ")
BASE_CONFIG = dict(architecture="smolmoe_v1", dim=32, hidden_dim=48,
                   layers=2, heads=4, kv_heads=2, experts=8, top_k=2,
                   vocab_size=260, max_context=512, shared_experts=1,
                   bos_id=256, eos_id=257, rms_eps=1e-5, rope_theta=10000)


def write_config(path, config):
    Path(path).write_text("".join(f"{k}={v}\n" for k, v in config.items()))


def read_config(path):
    ini = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    ini.read_string("[model]\n" + Path(path).read_text())
    raw = dict(ini["model"])
    if set(raw) != set(BASE_CONFIG):
        raise ValueError("config must have exactly the keys in examples/config.ini")
    c = {k: (v if k == "architecture" else float(v) if k in ("rms_eps", "rope_theta") else int(v))
         for k, v in raw.items()}
    d, h = c["dim"], c["hidden_dim"]
    if (c["architecture"] not in ("smolmoe_v1", "smolmoe_qwen2") or not 0 < d <= 65536 or not 0 < h <= 65536
            or not 0 < c["layers"] <= 1024 or not 0 < c["heads"] <= d
            or d % c["heads"] or (d // c["heads"]) % 2
            or not 0 < c["kv_heads"] <= c["heads"] or c["heads"] % c["kv_heads"]
            or not 0 < c["experts"] <= 4096 or not 0 < c["top_k"] <= c["experts"]
            or not 0 <= c["shared_experts"] <= 16 or not 258 <= c["vocab_size"] <= 262402
            or not 0 < c["max_context"] <= 1048576 or c["bos_id"] != 256 or c["eos_id"] != 257
            or not 0 < c["rms_eps"] < 1 or not 1 < c["rope_theta"] < float("inf")):
        raise ValueError("invalid/unsupported config")
    return c


def shapes(c):
    """name -> (rows, columns, output dtype: 0=F32, 1=Q8)."""
    d, h, v = c["dim"], c["hidden_dim"], c["vocab_size"]
    kd = c["kv_heads"] * (d // c["heads"])
    result = {"embed": (v, d, 1), "final_norm": (1, d, 0), "lm_head": (v, d, 1)}
    for layer in range(c["layers"]):
        p = f"l.{layer}."
        result.update({p + "attn_norm": (1, d, 0), p + "ffn_norm": (1, d, 0),
                       p + "q": (d, d, 1), p + "k": (kd, d, 1), p + "v": (kd, d, 1),
                       p + "o": (d, d, 1), p + "router": (c["experts"], d, 1)})
        if c["architecture"] == "smolmoe_qwen2":
            result.update({p + "q_bias": (1, d, 0), p + "k_bias": (1, kd, 0), p + "v_bias": (1, kd, 0)})
        for kind, count in (("e", c["experts"]), ("s", c["shared_experts"])):
            for e in range(count):
                result[p + f"{kind}.{e}.gate"] = (h, d, 1)
                result[p + f"{kind}.{e}.up"] = (h, d, 1)
                result[p + f"{kind}.{e}.down"] = (d, h, 1)
    return dict(sorted(result.items()))


def q8_row(values):
    data = bytearray()
    for i in range(0, len(values), 32):
        block = values[i:i + 32]
        if not all(math.isfinite(v) for v in block):
            raise ValueError("non-finite source weight")
        scale = max(map(abs, block), default=0) / 127
        scale = struct.unpack("<f", struct.pack("<f", scale))[0]
        q = [max(-127, min(127, round(v / scale))) if scale else 0 for v in block]
        q.extend([0] * (32 - len(q)))
        data.extend(struct.pack("<f32b", scale, *q))
    return bytes(data)


def write_checkpoint(output, specs, row_source, gap=0, aliases=None):
    """Stream rows, then install a sorted tensor directory. gap creates sparse
    address-space fixtures; never a claim of trained multi-gigabyte weights."""
    output = Path(output)
    tmp = output.with_suffix(output.suffix + ".tmp")
    entries, extents = [], {}
    aliases = aliases or {}
    try:
        with tmp.open("wb") as f:
            f.write((MAGIC[:-1] + b"\2" if aliases else MAGIC) + struct.pack("<II", len(specs), 0))
            f.seek(16 + RECORD.size * len(specs) + gap)
            for name, (rows, cols, dtype) in specs.items():
                if name in aliases:
                    if name != "lm_head" or aliases[name] != "embed" or specs[name] != specs["embed"]:
                        raise ValueError("only matching lm_head->embed alias is allowed")
                    offset, size = extents["embed"]
                    entries.append(RECORD.pack(name.encode(), rows, cols, dtype, 1, offset, size))
                    continue
                offset = f.tell()
                for row in row_source(name, rows, cols):
                    if len(row) != cols or not all(math.isfinite(x) for x in row):
                        raise ValueError(f"invalid F32 row in {name}")
                    f.write(q8_row(row) if dtype else struct.pack(f"<{cols}f", *row))
                size = f.tell() - offset
                expected = rows * (((cols + 31) // 32) * 36 if dtype else cols * 4)
                if size != expected:
                    raise ValueError(f"row count mismatch: {name}")
                encoded = name.encode("ascii")
                if len(encoded) >= 40:
                    raise ValueError("tensor name too long")
                entries.append(RECORD.pack(encoded, rows, cols, dtype, 0, offset, size))
                extents[name] = (offset, size)
            f.seek(16)
            f.write(b"".join(entries))
        tmp.replace(output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def fixture(directory, config=None, seed=7, gap=0, echo=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    c = dict(BASE_CONFIG if config is None else config)
    if echo:
        # A deliberately constructed transition circuit. NOT a trained LLM.
        c.update(dim=32, layers=1, hidden_dim=32, vocab_size=258,
                 heads=4, kv_heads=2, experts=4, top_k=2)
    write_config(directory / "config.ini", c)
    (directory / "tokenizer.bpe").write_text("SMBPE1\n" + ("72 105\n258 33\n" if c["vocab_size"] == 260 else ""))
    rng = random.Random(seed)
    if echo:
        # A cycle with unique input symbols; this circuit is not language.
        phrase = b"CPU works!\n"
        next_byte = {phrase[i]: phrase[(i + 1) % len(phrase)] for i in range(len(phrase))}
        channels = {byte: i for i, byte in enumerate(sorted(set(phrase)))}
    def rows(name, nr, nc):
        for r in range(nr):
            if name.endswith("norm"):
                yield [1.0] * nc
            elif echo:
                values = [0.0] * nc
                if name == "embed":
                    if r in channels:
                        values[channels[r]] = 1
                    else:
                        values[-1] = 1
                elif name == "lm_head":
                    for byte, nxt in next_byte.items():
                        if r == nxt:
                            values[channels[byte]] = 1
                    if r == phrase[0]:
                        values[-1] = 1
                yield values
            else:
                scale = 0.4 / math.sqrt(nc)
                values = [rng.uniform(-scale, scale) for _ in range(nc)]
                if name == "lm_head" and r in (256, 257):
                    values = [0.0] * nc
                yield values
    write_checkpoint(directory / "model.smoe", shapes(c), rows, gap)
    return c


def pack(source, output):
    """Manifest maps canonical tensor name to {file, offset?}; source weights
    are contiguous little-endian F32, row-major. Source rows stay bounded."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("source and output directories must differ")
    c = read_config(source / "config.ini")
    manifest = json.loads((source / "manifest.json").read_text())
    specs = shapes(c)
    if set(manifest) != set(specs):
        raise ValueError(f"tensor names mismatch: missing={set(specs)-set(manifest)}, extra={set(manifest)-set(specs)}")
    output.mkdir(parents=True, exist_ok=True)
    def rows(name, nr, nc):
        entry = manifest[name]
        offset = entry.get("offset", 0)
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("invalid F32 source offset")
        with (source / entry["file"]).open("rb") as f:
            f.seek(offset)
            for _ in range(nr):
                raw = f.read(nc * 4)
                if len(raw) != nc * 4:
                    raise ValueError(f"truncated source tensor: {name}")
                yield struct.unpack(f"<{nc}f", raw)
    write_checkpoint(output / "model.smoe", specs, rows)
    shutil.copyfile(source / "config.ini", output / "config.ini")
    shutil.copyfile(source / "tokenizer.bpe", output / "tokenizer.bpe")


def inspect(path):
    with Path(path).open("rb") as f:
        if f.read(8) not in (MAGIC, MAGIC[:-1]+b"\2"):
            raise ValueError("bad checkpoint magic")
        count, reserved = struct.unpack("<II", f.read(8))
        if reserved or not 0 < count <= 2000000:
            raise ValueError("invalid directory")
        entries = []
        for _ in range(count):
            name, rows, cols, dtype, _, off, size = RECORD.unpack(f.read(RECORD.size))
            entries.append(dict(name=name.split(b"\0")[0].decode(), rows=rows, cols=cols,
                                dtype=dtype, offset=off, bytes=size))
        return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fixture", help="create SYNTHETIC test weights, not a trained model")
    f.add_argument("directory")
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--echo", action="store_true", help="handcrafted byte transition circuit")
    f.add_argument("--sparse-gap-gib", type=int, default=0)
    k = sub.add_parser("pack", help="convert canonical raw F32 manifest, row by row")
    k.add_argument("source"); k.add_argument("output")
    i = sub.add_parser("inspect"); i.add_argument("checkpoint")
    a = p.parse_args()
    try:
        if a.command == "fixture":
            if a.sparse_gap_gib < 0:
                raise ValueError("negative sparse gap")
            fixture(a.directory, seed=a.seed, gap=a.sparse_gap_gib << 30, echo=a.echo)
            print("Created synthetic inference fixture (not trained language weights).")
        elif a.command == "pack":
            pack(a.source, a.output)
        else:
            print(json.dumps(inspect(a.checkpoint), indent=2))
    except (ValueError, OSError, KeyError, struct.error) as e:
        p.exit(1, f"checkpoint: {e}\n")


if __name__ == "__main__":
    main()
