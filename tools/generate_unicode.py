#!/usr/bin/env python3
"""Regenerate compact C Unicode classification and NFC tables using Python UCD.
Bundled tables use Unicode 15.0.0 (Python 3.12); no Python needed at inference.
"""
from pathlib import Path
import unicodedata as u


def generate(output):
    ranges=[];start=0;old=None
    for cp in range(0x110000):
        cat=u.category(chr(cp));value=1 if cat.startswith("L") else 2 if cat.startswith("N") else 0
        if value!=old:
            if old:ranges.append((start,cp-1,old))
            start,old=cp,value
    if old:ranges.append((start,0x10ffff,old))
    comb=[];decomp=[];compose=[]
    for cp in range(0x110000):
        cc=u.combining(chr(cp))
        if cc:comb.append((cp,cc))
        dec=u.decomposition(chr(cp))
        if dec and not dec.startswith("<"):
            ds=[int(x,16) for x in dec.split()]
            if len(ds)>2:raise ValueError("unexpected canonical decomposition")
            decomp.append((cp,ds[0],ds[1] if len(ds)>1 else 0))
            if len(ds)==2 and u.normalize("NFC","".join(map(chr,ds)))==chr(cp):compose.append((ds[0],ds[1],cp))
    lines=["/* Generated from Python unicodedata "+u.unidata_version+". No runtime Unicode dependency. */",
           "typedef struct {uint32_t a,b,value;} UnicodeRange;","static const UnicodeRange unicode_class[]={"]
    lines += ["{%d,%d,%d},"%x for x in ranges]
    lines += ["};","typedef struct {uint32_t cp,value;} UnicodeCCC;","static const UnicodeCCC unicode_ccc[]={"]
    lines += ["{%d,%d},"%x for x in comb]
    lines += ["};","static const UnicodeRange unicode_decomp[]={"]
    lines += ["{%d,%d,%d},"%x for x in decomp]
    lines += ["};","static const UnicodeRange unicode_compose[]={"]
    lines += ["{%d,%d,%d},"%x for x in sorted(compose)]
    Path(output).write_text("\n".join(lines+["};"])+"\n")
    print("Generated Unicode",u.unidata_version,"tables:",output)


if __name__=="__main__":generate(Path(__file__).resolve().parents[1]/"src/unicode_data.h")
