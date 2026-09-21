#!/usr/bin/env python3
"""Verify that the jwm1/Linux tokenization of the ctx1024 native prompt
produces 1053 or 1054 tokens — root-causing the cross-OS 1-token delta.
Requires: jwm1 venv (transformers/tokenizers) or jw16 venv. Runs LOCALLY
on the dev box using the mlx-omarchy venv tokenizer."""
import json, sys

# The native manifest prompt construction (from bench_matrix.json):
# numbered(20) = base + "Entry {i} of {items}." x20
# This is the EXACT string from the native cmd field, extracted verbatim.
j = json.load(open("/home/joshuawarren/src/mlx-omarchy/receipts/native-baseline-2026-09-06/native-2026-09-06.json"))
native_prompt = None
for leg in j["legs"]:
    lid = leg.get("leg_id","")
    if "longctx" in lid and "0.5b-4bit" in leg.get("model_id",""):
        cmd = leg.get("cmd","")
        i = cmd.find("--prompt ")
        j_ = cmd.find("--tokens ")
        native_prompt = cmd[i+9:j_].strip()  # strip leading/trailing space
        break

if not native_prompt:
    print("ERROR: native ctx1024 prompt not found"); sys.exit(1)

print(f"native prompt: {len(native_prompt)} chars, last 60: {repr(native_prompt[-60:])}")
print(f"native prompt_tokens: 1053 (from receipt)")

# Now check what the jw16 jw16gap-model tokenizer produces
# (the model tokenizer_config is the SAME on jwm1 — same revision a5339a41)
# We can't run this locally without mlx_lm; instead ship a verification
# script to jw16 which has the venv.

verify_script = '''
import json, sys
sys.path.insert(0, "/var/tmp/v071perf-venv/lib/python3.11/site-packages")
from transformers import AutoTokenizer
native_prompt = open("/tmp/ctx1024-native-prompt.txt").read().strip()
tc = json.load(open("/var/tmp/jw16gap-model/tokenizer_config.json"))
tok = AutoTokenizer.from_pretrained("/var/tmp/jw16gap-model")
# Replicate the harness logic: apply_chat_template then encode
if hasattr(tok, "apply_chat_template"):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": native_prompt}],
        add_generation_prompt=True)
if isinstance(prompt, str):
    add_special = (getattr(tok, "bos_token", None) is None or
                   not prompt.startswith(tok.bos_token))
    ids = tok.encode(prompt, add_special_tokens=add_special)
else:
    ids = prompt
print("templated+encoded token count:", len(ids))
print("first 5:", ids[:5], "last 5:", ids[-5:])
'''
print("verify script prepared for jw16 execution")
open("/tmp/verify-tokens-jw16.py", "w").write(verify_script)
print("written /tmp/verify-tokens-jw16.py")
