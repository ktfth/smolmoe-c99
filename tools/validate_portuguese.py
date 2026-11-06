#!/usr/bin/env python3
"""Run Portuguese prompts locally and compare memory-budget logit hashes.
No external inference service, tokenizer library or ML framework is used.
"""
import argparse
import json
from pathlib import Path
import subprocess


def run(root,d,prompt,memory=256,scalar=False,hashed=False,system=None,tokens=64,context=512):
    args=[str(root/"smolmoe"),"--model",str(d/"model.smoe"),"--config",str(d/"config.ini"),"--tokenizer",str(d/"tokenizer.qbp"),"--chat","--prompt",prompt,"--tokens",str(tokens),"--context",str(context),"--memory-mib",str(memory)]
    if scalar:args += ["--scalar"]
    if hashed:args += ["--hash-logits"]
    if system is not None:args += ["--system",system]
    p=subprocess.run(args,capture_output=True,check=True,timeout=120)
    return {"prompt":prompt,"response":p.stdout.decode("utf-8"),"stats":json.loads(p.stderr)}


def validate(directory,root):
    cases=["Escreva uma saudação curta para alguém que está aprendendo programação.","Quanto é 7 mais 5? Responda apenas com o número.","Escreva duas frases sobre um robô que cuida de uma horta."]
    results=[]
    for prompt in cases:
        item=run(root,directory,prompt);results.append(item);print(json.dumps(item,ensure_ascii=False),flush=True)
    (root/"docs/resultados-portugues.json").write_text(json.dumps({"system":"Responda somente em português do Brasil. Não misture outros idiomas. Seja breve e direto.","cases":results},ensure_ascii=False,indent=2)+"\n")
    observations=[];expected=None
    for budget,scalar in ((128,False),(256,False),(512,False),(128,True)):
        item=run(root,directory,cases[1],budget,scalar,True,"Responda em português.",8,256)
        actual=(item["response"],item["stats"]["logits_hash"])
        if expected is None:expected=actual
        if expected!=actual:raise AssertionError("budget/backend text or raw-logit hash mismatch")
        if item["stats"]["tracked_peak_bytes"]>budget*1048576:raise AssertionError("budget exceeded")
        observations.append(dict(budget_mib=budget,**item))
    (root/"docs/qwen-memory-validation.json").write_text(json.dumps({"passed":len(observations),"prompt":cases[1],"system":"Responda em português.","hash":"FNV-1a64 of every raw F32 logit in little-endian token/position order","runs":observations},ensure_ascii=False,indent=2)+"\n")
    print("Portuguese execution and budget/hash comparisons passed.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path,nargs="?",default=Path("examples/qwen-portugues"));a=p.parse_args()
    validate(a.directory.resolve(),Path(__file__).resolve().parents[1])
