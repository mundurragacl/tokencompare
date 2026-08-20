#!/usr/bin/env python3
"""Produce ground-truth token counts using the reference implementations.

HF `tokenizers` for tokenizer.json models, `tiktoken` for OpenAI encodings and
for Kimi's tiktoken.model. Output is written as JSON so the JS implementation
can be diffed against it.
"""
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)

CORPUS = json.loads((ROOT / "test" / "corpus.json").read_text())

# Include the published fixtures, so verification covers the exact inputs the
# README quotes numbers for, not just short synthetic cases.
for name, path in [("fixture:prose", "fixtures/prose.txt"), ("fixture:code", "fixtures/code.js")]:
    f = ROOT / path
    if f.exists():
        CORPUS.append({"name": name, "text": f.read_text()})

# Same pattern string as moonshotai/Kimi-K3 tokenization_kimi.py.
KIMI_PAT = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


def fetch(repo: str, filename: str) -> Path:
    dest = CACHE / f"{repo.replace('/', '__')}.{filename}"
    if not dest.exists():
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        print(f"  downloading {repo}/{filename}")
        urllib.request.urlretrieve(url, dest)
    return dest


results = {}

# --- HF tokenizer.json models -------------------------------------------------
from tokenizers import Tokenizer

for repo in ["zai-org/GLM-5.2", "deepseek-ai/DeepSeek-V4-Pro"]:
    print(f"loading {repo}")
    tok = Tokenizer.from_file(str(fetch(repo, "tokenizer.json")))
    results[repo] = {
        c["name"]: len(tok.encode(c["text"], add_special_tokens=False).ids)
        for c in CORPUS
    }

# --- OpenAI encodings ---------------------------------------------------------
import tiktoken

# encode_ordinary, not encode: special-token literals in the input are counted as
# ordinary text rather than raising, which is what the JS path does too.
enc = tiktoken.get_encoding("o200k_base")
results["o200k_base"] = {c["name"]: len(enc.encode_ordinary(c["text"])) for c in CORPUS}

# --- Kimi (tiktoken.model) ----------------------------------------------------
for repo in ["moonshotai/Kimi-K3", "moonshotai/Kimi-K2.6"]:
    print(f"loading {repo}")
    path = fetch(repo, "tiktoken.model")
    ranks = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    kenc = tiktoken.Encoding(
        name=repo, pat_str=KIMI_PAT, mergeable_ranks=ranks, special_tokens={}
    )
    results[repo] = {c["name"]: len(kenc.encode(c["text"])) for c in CORPUS}
    results[repo + "__meta"] = {"vocab_size": len(ranks)}

out = ROOT / "test" / "reference.json"
out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
print(f"\nwrote {out}")
for k, v in results.items():
    if k.endswith("__meta"):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: total={sum(v.values())}")
