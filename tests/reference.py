"""Independent, slow float64 decoder for the supported architecture.
Uses stdlib only; dequantizes fixtures solely for numerical verification.
Does not reuse production C kernels, routing, attention or checkpoint writer.
"""
import math
import struct
from pathlib import Path


def load(path):
    with Path(path).open("rb") as f:
        assert f.read(8) in (b"SMOEQ8\0\1",b"SMOEQ8\0\2")
        count, reserved = struct.unpack("<II", f.read(8))
        assert not reserved
        entries = []
        for _ in range(count):
            name, rows, cols, kind, _, offset, size = struct.unpack("<40sIIIIQQ", f.read(72))
            entries.append((name.split(b"\0")[0].decode(), rows, cols, kind, offset, size))
        result = {}
        for name, rows, cols, kind, offset, _ in entries:
            f.seek(offset)
            matrix = []
            for _ in range(rows):
                if not kind:
                    row = list(struct.unpack(f"<{cols}f", f.read(cols * 4)))
                else:
                    row = []
                    for _ in range((cols + 31) // 32):
                        scale, *q = struct.unpack("<f32b", f.read(36))
                        row.extend(scale * v for v in q)
                    row = row[:cols]
                matrix.append(row)
            result[name] = matrix
        return result


def mv(a, x):
    return [sum(v * w for v, w in zip(row, x)) for row in a]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def rms(x, weights, epsilon):
    factor = 1 / math.sqrt(sum(v * v for v in x) / len(x) + epsilon)
    return [v * factor * w for v, w in zip(x, weights[0])]


def rotate(x, head_dim, position, theta):
    out = list(x)
    for h in range(len(x) // head_dim):
        for j in range(0, head_dim, 2):
            angle = position / theta ** (j / head_dim)
            a, b = x[h * head_dim + j:h * head_dim + j + 2]
            out[h * head_dim + j] = a * math.cos(angle) - b * math.sin(angle)
            out[h * head_dim + j + 1] = a * math.sin(angle) + b * math.cos(angle)
    return out


def run(c, weights, tokens):
    keys, values = [[] for _ in range(c["layers"])], [[] for _ in range(c["layers"])]
    hd = c["dim"] // c["heads"]
    result = []
    for position, token in enumerate(tokens):
        x = weights["embed"][token][:]
        for layer in range(c["layers"]):
            p = f"l.{layer}."
            n = rms(x, weights[p + "attn_norm"], c["rms_eps"])
            q, k, v = mv(weights[p + "q"], n),mv(weights[p + "k"], n),mv(weights[p + "v"], n)
            if c["architecture"] == "smolmoe_qwen2":
                q,k,v=add(q,weights[p+"q_bias"][0]),add(k,weights[p+"k_bias"][0]),add(v,weights[p+"v_bias"][0])
            q = rotate(q, hd, position, c["rope_theta"])
            k = rotate(k, hd, position, c["rope_theta"])
            keys[layer].append(k)
            values[layer].append(v)
            att = []
            for h in range(c["heads"]):
                kh = h // (c["heads"] // c["kv_heads"])
                scores = [sum(q[h * hd + j] * old[kh * hd + j] for j in range(hd)) / math.sqrt(hd)
                          for old in keys[layer]]
                probs = [math.exp(s - max(scores)) for s in scores]
                total = sum(probs)
                att.extend(sum(prob / total * v[kh * hd + j] for prob, v in zip(probs, values[layer]))
                           for j in range(hd))
            x = add(x, mv(weights[p + "o"], att))
            n = rms(x, weights[p + "ffn_norm"], c["rms_eps"])
            route = mv(weights[p + "router"], n)
            selected = sorted(range(c["experts"]), key=lambda i: (-route[i], i))[:c["top_k"]]
            probs = [1 / (1 + math.exp(-route[i])) for i in selected]
            total = sum(probs)
            def expert(kind, i):
                ep = p + f"{kind}.{i}."
                g, up = mv(weights[ep + "gate"], n), mv(weights[ep + "up"], n)
                return mv(weights[ep + "down"], [u * v / (1 + math.exp(-v)) for u, v in zip(up, g)])
            for i, prob in zip(selected, probs):
                x = add(x, [v * prob / total for v in expert("e", i)])
            for i in range(c["shared_experts"]):
                x = add(x, expert("s", i))
        result.append(mv(weights["lm_head"], rms(x, weights["final_norm"], c["rms_eps"])))
    return result
