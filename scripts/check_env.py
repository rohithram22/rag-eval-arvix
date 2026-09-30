"""Step 1 sanity check: packages, config, API keys, LLM round-trips, embeddings.

Usage:
    python scripts/check_env.py                # run all checks
    python scripts/check_env.py --list-models  # list model IDs your keys can use
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

FAILURES: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    FAILURES.append(msg)


def check_packages() -> None:
    print("\n1) Python & packages")
    print(f"  Python {sys.version.split()[0]}  (venv active: {sys.prefix != sys.base_prefix})")
    modules = [
        "llama_index.core", "llama_index.embeddings.huggingface",
        "llama_index.vector_stores.faiss", "llama_index.retrievers.bm25",
        "llama_index.llms.groq", "llama_index.llms.google_genai",
        "faiss", "sentence_transformers", "torch", "fitz", "arxiv",
        "yaml", "pandas", "streamlit", "rag_eval",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
            ok(mod)
        except Exception as e:  # noqa: BLE001
            fail(f"import {mod}: {type(e).__name__}: {e}")
    try:
        import torch

        print(f"  torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} (expected False)")
    except Exception:  # noqa: BLE001
        pass


def check_config() -> dict | None:
    print("\n2) Config")
    try:
        from rag_eval.config import config_hash, load_config

        cfg = load_config()
        h1 = config_hash(cfg, ["chunking", "embedding"])
        ok(f"base.yaml loaded | chunk_size={cfg['chunking']['chunk_size']} | index hash={h1}")

        cfg2 = load_config(overrides={"chunking.chunk_size": 256})
        h2 = config_hash(cfg2, ["chunking", "embedding"])
        if h1 != h2 and cfg2["chunking"]["chunk_size"] == 256:
            ok(f"override applied | chunk_size=256 -> index hash={h2} (different, as it should be)")
        else:
            fail("override did not change the config hash")

        try:
            load_config(overrides={"chunking.chunksize": 256})
            fail("typo'd override was silently accepted")
        except KeyError:
            ok("typo'd override 'chunking.chunksize' correctly rejected")
        return cfg
    except Exception as e:  # noqa: BLE001
        fail(f"config: {type(e).__name__}: {e}")
        return None


def check_keys() -> None:
    print("\n3) API keys")
    for name in ["GROQ_API_KEY", "GOOGLE_API_KEY"]:
        if os.environ.get(name):
            ok(f"{name} is set")
        else:
            fail(f"{name} missing (add Codespaces secret + reload window, or use .env)")


def check_llms(cfg: dict) -> None:
    print("\n4) LLM round-trips")
    from rag_eval.llms import get_llm

    for role in ["generator", "qgen", "judge"]:
        section = cfg[role]
        label = f"{role:<9} {section['provider']}/{section['model']}"
        try:
            llm = get_llm(section)
            t0 = time.time()
            resp = llm.complete("Reply with exactly one word: pong")
            text = str(resp).strip().replace("\n", " ")
            if text:
                ok(f"{label} -> {text[:40]!r} ({time.time() - t0:.1f}s)")
            else:
                fail(f"{label} returned empty text (try a larger max_tokens)")
        except Exception as e:  # noqa: BLE001
            fail(f"{label}: {type(e).__name__}: {str(e)[:200]}")


def check_embeddings(cfg: dict) -> None:
    print("\n5) Embedding model (first run downloads ~130 MB)")
    try:
        import numpy as np
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        emb = HuggingFaceEmbedding(model_name=cfg["embedding"]["model_name"])
        texts = [
            "Dense retrieval encodes queries and passages into vectors.",
            "Vector search returns passages whose embeddings are closest to the query embedding.",
            "The recipe calls for two cups of flour and a pinch of salt.",
        ]
        vecs = np.array([emb.get_text_embedding(t) for t in texts])
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        sims = vecs @ vecs.T
        print(f"  dim={vecs.shape[1]} | cos(retrieval, retrieval)={sims[0, 1]:.3f} | "
              f"cos(retrieval, recipe)={sims[0, 2]:.3f}")
        if sims[0, 1] > sims[0, 2]:
            ok("related sentences are closer than unrelated ones")
        else:
            fail("embedding similarity ordering looks wrong")
    except Exception as e:  # noqa: BLE001
        fail(f"embeddings: {type(e).__name__}: {e}")


def list_models() -> None:
    import httpx
    from rag_eval import config  # noqa: F401  (loads .env)

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        r = httpx.get("https://api.groq.com/openai/v1/models",
                      headers={"Authorization": f"Bearer {groq_key}"}, timeout=30)
        r.raise_for_status()
        ids = sorted(m["id"] for m in r.json()["data"])
        print("Groq models:\n  " + "\n  ".join(ids))
    else:
        print("GROQ_API_KEY not set; skipping Groq.")

    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models",
                      params={"key": google_key, "pageSize": 200}, timeout=30)
        r.raise_for_status()
        ids = sorted(
            m["name"].removeprefix("models/") for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        )
        print("\nGemini models (generateContent):\n  " + "\n  ".join(ids))
    else:
        print("GOOGLE_API_KEY not set; skipping Gemini.")
    print("\nCheck free-tier limits: console.groq.com/settings/limits and aistudio.google.com")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return 0

    check_packages()
    cfg = check_config()
    check_keys()
    if cfg is not None:
        check_llms(cfg)
        check_embeddings(cfg)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
