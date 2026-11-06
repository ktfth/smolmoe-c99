#!/usr/bin/env python3
"""Import explicitly supported Qwen2.5-0.5B-Instruct BF16 Safetensors.
Stdlib metadata/conversion orchestration; row conversion is C99. No framework.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import urllib.request

import checkpoint as ck
from import_llama2c import sha256, download

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2"
MODEL_HASH = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"


def byte_alphabet():
    byte_ids=list(range(33,127))+list(range(161,173))+list(range(174,256))
    chars=list(byte_ids)
    for b in range(256):
        if b not in byte_ids:
            byte_ids.append(b);chars.append(256+len(chars)-188)
    return {b:chr(ch) for b,ch in zip(byte_ids,chars)}


def export_tokenizer(source, output, vocab):
    t=json.loads(Path(source).read_text());base=t["model"]["vocab"]
    if t["normalizer"]!={"type":"NFC"} or t["model"]["type"]!="BPE":raise ValueError("unsupported tokenizer")
    alphabet=byte_alphabet();inverse={v:k for k,v in alphabet.items()}
    old=[base[alphabet[b]] for b in range(256)]+[151643,151645]
    used=set(old);old += [i for i in range(vocab) if i not in used]
    to_new={o:n for n,o in enumerate(old)}
    pieces=[b""]*vocab;special=[1]*vocab
    for piece,oid in base.items():pieces[to_new[oid]]=bytes(inverse[c] for c in piece);special[to_new[oid]]=0
    for item in t["added_tokens"]:
        pieces[to_new[item["id"]]]=item["content"].encode();special[to_new[item["id"]]]=1
    for n,p in enumerate(pieces):
        if not p:pieces[n]=f"<unused:{old[n]}>".encode()
    merges=[]
    for pair in t["model"]["merges"]:
        a,b=pair.split(" ") if isinstance(pair,str) else pair
        merges.append((to_new[base[a]],to_new[base[b]],to_new[base[a+b]]))
    with Path(output).open("wb") as f:
        f.write(b"SMQBP1\0\1"+struct.pack("<IIII",vocab,len(merges),sum(map(len,pieces)),to_new[151644]))
        for p,s in zip(pieces,special):f.write(struct.pack("<II",len(p),s)+p)
        for a,b,o in merges:f.write(struct.pack("<III",a,b,o))
    return old,to_new


def convert(source,output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=json.loads((source/"config.json").read_text())
    if(raw.get("model_type")!="qwen2" or raw.get("use_sliding_window") or raw.get("rope_scaling") or raw.get("hidden_act")!="silu" or not raw.get("tie_word_embeddings")):
        raise ValueError("only full-attention, tied-head Qwen2 SiLU supported")
    c=dict(ck.BASE_CONFIG,architecture="smolmoe_qwen2",dim=raw["hidden_size"],hidden_dim=raw["intermediate_size"],
           layers=raw["num_hidden_layers"],heads=raw["num_attention_heads"],kv_heads=raw["num_key_value_heads"],
           experts=1,top_k=1,shared_experts=0,vocab_size=raw["vocab_size"],max_context=raw["max_position_embeddings"],
           rms_eps=raw["rms_norm_eps"],rope_theta=raw["rope_theta"])
    ck.write_config(output/"config.ini",c);ck.read_config(output/"config.ini")
    old,to_new=export_tokenizer(source/"tokenizer.json",output/"tokenizer.qbp",c["vocab_size"])
    checkpoint=source/"model.safetensors"
    if sha256(checkpoint)!=MODEL_HASH:raise ValueError("official model checksum mismatch")
    with checkpoint.open("rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        if n>16<<20:raise ValueError("header too large")
        tensors=json.loads(f.read(n));data_start=8+n
    tool=Path(__file__).with_name("q8pack")
    subprocess.run(["cc","-O3","-std=c99","-Wall","-Wextra","-Werror","-ffp-contract=off",str(tool.with_suffix(".c")),"-o",str(tool),"-lm"],check=True)
    specs=ck.shapes(c);entries=[];extents={};hd=c["dim"]//c["heads"]
    with tempfile.TemporaryDirectory() as temp:
        row_map=Path(temp)/"rows.bin";row_map.write_bytes(struct.pack(f"<{len(old)}I",*old))
        target=output/"model.smoe";partial=target.with_suffix(".partial")
        with partial.open("wb") as out:
            out.write(b"SMOEQ8\0\2"+struct.pack("<II",len(specs),0));out.seek(16+len(specs)*72)
            for name,(rows,cols,dtype) in specs.items():
                off=out.tell();flags=0
                if name=="lm_head":off,size=extents["embed"];flags=1
                elif name.endswith("router"):
                    out.write(ck.q8_row([0.0]*cols));size=out.tell()-off
                else:
                    perm=0
                    if name=="embed":key="model.embed_tokens.weight"
                    elif name=="final_norm":key="model.norm.weight"
                    else:
                        _,layer,*tail=name.split(".");suffix=tail[-1];p=f"model.layers.{layer}."
                        if suffix=="attn_norm":key=p+"input_layernorm.weight"
                        elif suffix=="ffn_norm":key=p+"post_attention_layernorm.weight"
                        elif suffix in ("gate","up","down"):key=p+f"mlp.{suffix}_proj.weight"
                        else:
                            proj=suffix.split("_")[0];bias=suffix.endswith("_bias")
                            key=p+f"self_attn.{proj}_proj."+("bias" if bias else "weight")
                            if proj in ("q","k") and not bias:perm=hd
                    meta=tensors[key]
                    expected=[cols] if dtype==0 else [rows,cols]
                    if meta["shape"]!=expected or meta["dtype"]!="BF16":raise ValueError(f"unexpected tensor: {key}")
                    # Q/K bias has the same split-half -> adjacent permutation.
                    if name.endswith(("q_bias","k_bias")):
                        with checkpoint.open("rb") as f:f.seek(data_start+meta["data_offsets"][0]);b=f.read(cols*2)
                        ordered=b"".join(b[(head*hd+(i%2)*(hd//2)+i//2)*2:(head*hd+(i%2)*(hd//2)+i//2)*2+2] for head in range(cols//hd) for i in range(hd))
                        for (u,) in struct.iter_unpack("<H",ordered):out.write(struct.pack("<I",u<<16))
                    else:
                        out.flush()
                        args=[str(tool),str(checkpoint),str(data_start+meta["data_offsets"][0]),str(rows),str(cols),"1",str(dtype),str(perm)]
                        if name=="embed":args.append(str(row_map))
                        subprocess.run(args,stdout=out,check=True)
                    size=out.tell()-off
                    if size!=rows*(((cols+31)//32)*36 if dtype else cols*4):raise ValueError("converter size mismatch")
                extents[name]=(off,size);entries.append(ck.RECORD.pack(name.encode(),rows,cols,dtype,flags,off,size))
            out.seek(16);out.write(b"".join(entries))
        partial.replace(target)
    record=dict(model=REPO,revision=REVISION,source_sha256=MODEL_HASH,tokenizer_sha256=sha256(source/"tokenizer.json"),
                quantization="lossy Q8 block32",qk_permutation="split-half rotary to adjacent-pair, including bias",
                tied_lm_head=True,full_kimi_compatible=False)
    (output/"provenance.json").write_text(json.dumps(record,indent=2)+"\n")
    if(source/"LICENSE").exists():(output/"LICENSE-Qwen.txt").write_bytes((source/"LICENSE").read_bytes())
    return c


def fetch(source):
    source=Path(source);source.mkdir(parents=True,exist_ok=True)
    for name in ("config.json","tokenizer.json","LICENSE","model.safetensors"):
        path=source/name
        revision="main" if name=="LICENSE" else REVISION
        url=f"https://huggingface.co/{REPO}/resolve/{revision}/{name}"
        if name=="model.safetensors":download(url,path,MODEL_HASH)
        elif not path.exists():
            with urllib.request.urlopen(url,timeout=60) as r:path.write_bytes(r.read())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source");p.add_argument("output");p.add_argument("--fetch",action="store_true");a=p.parse_args()
    if a.fetch:fetch(a.source)
    convert(a.source,a.output)
    print("Converted Qwen2.5 to C99 Q8 format.")
