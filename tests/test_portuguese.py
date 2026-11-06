"""Offline checks: Qwen tokenizer goldens, Unicode, biased attention and aliasing."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unicodedata
import unittest

from test_runtime import ROOT
import checkpoint as ck
import reference


class PortugueseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.model = ROOT / "examples/qwen-portugues"

    def tearDown(self):
        self.tmp.cleanup()

    def tokenize(self, text, chat=False, system=None, tokenizer=None, ok=True):
        path, ids = self.work / "input", self.work / "ids"
        path.write_bytes(text.encode() if isinstance(text, str) else text)
        args = [str(ROOT / "smolmoe"), "--config", str(self.model / "config.ini"),
                "--tokenizer", str(tokenizer or self.model / "tokenizer.qbp"),
                "--prompt-file", str(path), "--context", "1024", "--tokenize-only", "--token-ids", str(ids)]
        if chat:args += ["--chat"]
        if system is not None:args += ["--system", system]
        p = subprocess.run(args, capture_output=True, timeout=30)
        self.assertEqual(p.returncode == 0, ok, p.stderr.decode(errors="replace"))
        return p.stdout, [int(x) for x in ids.read_text().splitlines()] if ok else []

    def test_upstream_tokenizer_goldens(self):
        records = json.loads((ROOT / "docs/qwen-tokenizer-validation.json").read_text())
        for case in records["cases"]:
            with self.subTest(case=case):
                chat = "chatml_prompt" in case
                text = case.get("text", case.get("chatml_prompt"))
                out, ids = self.tokenize(text, chat, case.get("system"))
                self.assertEqual(ids, case["runtime_ids"])
                if not chat:self.assertEqual(out, unicodedata.normalize("NFC",text).encode())

    def test_nfc_and_unicode_roundtrip(self):
        cases = ["ac\u0327a\u0303o e cafe\u0301", "\u1100\u1161\u11a8", "a\u0315\u0300", "\u212b\u0344", "\u1e0a\u0323", "\x00Olá\x00", "🌳\u200d🌱", "中文 123\n\n ", "'S 'RE 'VE 'LL\t\u00a0"]
        for text in cases:
            out, _ = self.tokenize(text)
            self.assertEqual(out,unicodedata.normalize("NFC",text).encode())

    def test_invalid_utf8_rejected(self):
        for text in (b"\xc0\x80",b"\xed\xa0\x80",b"\xf4\x90\x80\x80",b"\xe2\x82",b"\x80"):
            self.tokenize(text,ok=False)

    def test_corrupt_qwen_header_rejected(self):
        path = self.work / "bad.qbp"
        raw = (self.model / "tokenizer.qbp").read_bytes()
        path.write_bytes(raw[:8] + struct.pack("<I",1) + raw[12:])
        self.tokenize("Olá",tokenizer=path,ok=False)

    def synthetic(self, alias=False):
        directory = self.work / "synthetic"
        c = dict(ck.BASE_CONFIG,architecture="smolmoe_qwen2",dim=8,hidden_dim=12,layers=1,heads=2,kv_heads=1,experts=1,top_k=1,shared_experts=0)
        ck.fixture(directory,c)
        if alias:
            w=reference.load(directory/"model.smoe")
            ck.write_checkpoint(directory/"model.smoe",ck.shapes(c),lambda name,r,col:iter(w[name]),aliases={"lm_head":"embed"})
        return directory,c

    def test_biased_attention_and_tied_head_reference(self):
        for alias in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                # independent directory per case
                original=self.work;self.work=Path(tmp)
                d,c=self.synthetic(alias);trace=d/"trace"
                p=subprocess.run([str(ROOT/"smolmoe"),"--model",str(d/"model.smoe"),"--config",str(d/"config.ini"),"--tokenizer",str(d/"tokenizer.bpe"),"--prompt","Hi!","--tokens","0","--trace",str(trace),"--memory-mib","2"],capture_output=True,check=True)
                actual=[[float.fromhex(x) for x in line.split()[1:]] for line in trace.read_text().splitlines()]
                expected=reference.run(c,reference.load(d/"model.smoe"),[256,259])
                for a,b in zip(actual,expected):self.assertLess(max(abs(x-y) for x,y in zip(a,b)),2e-6)
                self.work=original

    def test_invalid_alias_rejected(self):
        d,_=self.synthetic(True)
        entries=ck.inspect(d/"model.smoe");index=next(i for i,e in enumerate(entries) if e["name"]=="lm_head")
        with(d/"model.smoe").open("r+b") as f:f.seek(16+index*72+56);f.write(struct.pack("<Q",entries[0]["offset"]+36))
        p=subprocess.run([str(ROOT/"smolmoe"),"--model",str(d/"model.smoe"),"--config",str(d/"config.ini"),"--tokenizer",str(d/"tokenizer.bpe"),"--prompt","Hi","--tokens","0"],capture_output=True)
        self.assertNotEqual(p.returncode,0);self.assertIn(b"alias",p.stderr)

    def test_c_bf16_row_converter_and_rotary_permutation(self):
        source=self.work/"source.bf16"
        values=[[float(r*8+i-10) for i in range(8)] for r in range(4)]
        source.write_bytes(b"".join(struct.pack("<H",struct.unpack("<I",struct.pack("<f",v))[0]>>16) for row in values for v in row))
        p=subprocess.run([str(ROOT/"tools/q8pack"),str(source),"0","4","8","1","1","4"],capture_output=True,check=True)
        for r,old in enumerate((0,2,1,3)):
            scale,*q=struct.unpack_from("<f32b",p.stdout,r*36)
            for x,y in zip(values[old],q):self.assertLessEqual(abs(x-y*scale),scale/2+1e-6)

    def test_prefill_skip_and_hash_trace_paths_match_text(self):
        d,_=self.synthetic()
        base=[str(ROOT/"smolmoe"),"--model",str(d/"model.smoe"),"--config",str(d/"config.ini"),"--tokenizer",str(d/"tokenizer.bpe"),"--prompt","Hi!","--tokens","8","--memory-mib","2"]
        a=subprocess.run(base,capture_output=True,check=True)
        b=subprocess.run(base+["--hash-logits"],capture_output=True,check=True)
        self.assertEqual(a.stdout,b.stdout);self.assertNotEqual(json.loads(b.stderr)["logits_hash"],"0000000000000000")


if __name__=="__main__":unittest.main()
