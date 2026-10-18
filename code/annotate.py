"""LLM hierarchical annotation with cached responses.

Each item is labelled k times (temperature sampling) by an OpenAI-compatible
chat model. The model must return one label path (coarse -> fine) plus a
verbalised confidence per level. Node scores used downstream:
  vote(v) = share of the k samples whose path contains v
  verb(v) = mean verbalised confidence of v over samples that contain it (0 otherwise)
Responses are cached in JSONL so reruns cost nothing.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "llm_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def load_env():
    p = Path.home() / ".config" / "rp" / "llm.env"
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


PROVIDERS = {
    "deepseek": ("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "alt": ("ALT_LLM_BASE_URL", "ALT_LLM_API_KEY", None),
}


def build_prompt(text: str, tree: dict) -> str:
    lines = []
    for a, subs in tree.items():
        if isinstance(subs, dict):
            for b, leaves in subs.items():
                lines.append(f"{a} > {b} > " + " | ".join(leaves))
        else:
            lines.append(f"{a} > " + " | ".join(subs))
    depth = 3 if isinstance(next(iter(tree.values())), dict) else 2
    keys = ", ".join(f'"l{i+1}"' for i in range(depth))
    conf = ", ".join(f'"c{i+1}"' for i in range(depth))
    return (
        "Classify the text into exactly one path of the taxonomy below "
        "(coarse to fine). Use label names exactly as written.\n\n"
        "TAXONOMY (each line: level-1 > level-2" + (" > level-3 options" if depth == 3 else " options") + "):\n"
        + "\n".join(lines)
        + f"\n\nTEXT:\n{text[:1500]}\n\n"
        f"Answer with JSON only: {{{keys}, {conf}}} where c* is your probability "
        "(0-1) that the label at that level is correct."
    )


_lock = threading.Lock()


def call(provider: str, model: str | None, prompt: str, temperature: float, seed: int,
         max_retries: int = 6) -> str:
    base_k, key_k, default_model = PROVIDERS[provider]
    url = os.environ[base_k].rstrip("/") + "/chat/completions"
    body = {
        "model": model or default_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
    }
    if seed is not None:
        body["seed"] = seed
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=body, timeout=90,
                              headers={"Authorization": f"Bearer {os.environ[key_k]}"})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code in (400, 422) and "response_format" in body:
                body.pop("response_format")
                continue
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt + random.random())
    return ""


def parse(resp: str, depth: int):
    m = re.search(r"\{.*\}", resp, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    path = [str(d.get(f"l{i+1}", "")).strip() for i in range(depth)]
    confs = []
    for i in range(depth):
        try:
            confs.append(min(1.0, max(0.0, float(d.get(f"c{i+1}", 0)))))
        except (TypeError, ValueError):
            confs.append(0.0)
    return path, confs


def annotate(name: str, items: list[dict], tree: dict, provider: str, model: str | None,
             k: int, temperature: float, workers: int):
    tag = f"{name}_{provider}_{model or 'default'}_k{k}_t{temperature}"
    out = CACHE / f"{tag}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["id"], r["s"]))
    jobs = [(it, s) for it in items for s in range(k) if (it["id"], s) not in done]
    print(f"{tag}: {len(done)} cached, {len(jobs)} to run", flush=True)
    f = out.open("a")

    def work(job):
        it, s = job
        resp = call(provider, model, build_prompt(it["text"], tree), temperature, seed=s)
        rec = {"id": it["id"], "s": s, "resp": resp}
        with _lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(work, jobs)):
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    f.close()
    return out


if __name__ == "__main__":
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from datasets_local import load_dataset_items
    items, tree = load_dataset_items(a.dataset, a.n)
    annotate(a.dataset, items, tree, a.provider, a.model, a.k, a.temperature, a.workers)
