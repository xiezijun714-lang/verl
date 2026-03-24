"""
BrowseComp-Plus local BM25 retrieval server.

Indexes all documents (gold_docs + negative_docs) from the browsecomp-plus-processed
dataset and serves a /retrieve endpoint compatible with verl's SearchTool.

Usage:
    python browsecomp_retrieval_server.py \
        --data_dir /root/paddlejob/workspace/xzj/dataset/browsecomp-plus-processed \
        --host 127.0.0.1 --port 8000

Endpoint:
    POST /retrieve
    Body: {"queries": ["query1", "query2"], "topk": 3}
    Response: {"result": [[{"document": {"contents": "title\\nbody"}, "score": 0.9}, ...], ...]}
"""

import argparse
import asyncio
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi


def extract_title(text: str, url: str = "") -> str:
    """Extract title from markdown front matter or fall back to URL."""
    m = re.search(r"^---\s*\ntitle:\s*(.+?)\n", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    if url:
        # Use last path component of URL as title
        return url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    return "Unknown"


def build_corpus(data_dir: str):
    """Load all unique documents from browsecomp parquet files."""
    print("Loading parquet files ...", flush=True)
    dfs = []
    for split in ["train.parquet", "test.parquet"]:
        path = f"{data_dir}/{split}"
        try:
            dfs.append(pd.read_parquet(path))
        except FileNotFoundError:
            print(f"  Warning: {path} not found, skipping.", flush=True)

    if not dfs:
        raise RuntimeError(f"No parquet files found in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)

    seen = {}
    for col in ["gold_docs", "negative_docs", "evidence_docs"]:
        if col not in df.columns:
            continue
        for row_docs in df[col].dropna():
            if row_docs is None:
                continue
            for doc in row_docs:
                if not isinstance(doc, dict):
                    continue
                docid = str(doc.get("docid", ""))
                if docid and docid not in seen:
                    seen[docid] = doc

    print(f"  Loaded {len(seen)} unique documents.", flush=True)
    return seen


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class RetrieverState:
    def __init__(self, data_dir: str, index_cache: str = "/tmp/browsecomp_bm25_cache.pkl"):
        if index_cache and os.path.exists(index_cache):
            print(f"Loading BM25 index from cache: {index_cache} ...", flush=True)
            with open(index_cache, "rb") as f:
                cached = pickle.load(f)
            self.docids = cached["docids"]
            self.docs = cached["docs"]
            self.bm25 = cached["bm25"]
            self.contents = cached["contents"]
            print(f"  Loaded {len(self.docids)} documents from cache.", flush=True)
        else:
            doc_map = build_corpus(data_dir)
            self.docids = list(doc_map.keys())
            self.docs = [doc_map[d] for d in self.docids]

            print("Building BM25 index ...", flush=True)
            corpus_texts = [d.get("text", "") for d in self.docs]
            tokenized = [tokenize(t) for t in corpus_texts]
            self.bm25 = BM25Okapi(tokenized)

            # Pre-build contents field (first line = title, rest = body)
            self.contents = []
            for doc in self.docs:
                title = extract_title(doc.get("text", ""), doc.get("url", ""))
                body = doc.get("text", "")
                self.contents.append(f"{title}\n{body}")

            if index_cache:
                print(f"Saving index cache to {index_cache} ...", flush=True)
                with open(index_cache, "wb") as f:
                    pickle.dump({
                        "docids": self.docids,
                        "docs": self.docs,
                        "bm25": self.bm25,
                        "contents": self.contents,
                    }, f)

        print("Ready.", flush=True)

    def search(self, query: str, topk: int = 3):
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:topk]
        results = []
        for idx in top_indices:
            results.append({
                "document": {"contents": self.contents[idx]},
                "score": float(scores[idx]),
            })
        return results


# ---- Process pool workers (bypass GIL for CPU-bound BM25) ----

_worker_state = None


def _pool_init(cache_path):
    """Load BM25 index in each worker process."""
    global _worker_state
    with open(cache_path, "rb") as f:
        _worker_state = pickle.load(f)
    print(f"  [Pool worker PID {os.getpid()}] Loaded {len(_worker_state['docids'])} docs from cache.", flush=True)


def _pool_search(args):
    """Run BM25 search in a worker process (GIL-free parallelism)."""
    query, topk = args
    tokens = tokenize(query)
    scores = _worker_state["bm25"].get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:topk]
    return [
        {"document": {"contents": _worker_state["contents"][idx]}, "score": float(scores[idx])}
        for idx in top_indices
    ]


# ---- FastAPI app ----

app = FastAPI()
state: Optional[RetrieverState] = None
search_pool: Optional[ProcessPoolExecutor] = None


class RetrieveRequest(BaseModel):
    queries: List[str]
    topk: int = 3
    return_scores: bool = True


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    assert search_pool is not None
    loop = asyncio.get_event_loop()
    args = [(q, req.topk) for q in req.queries]
    futures = [loop.run_in_executor(search_pool, _pool_search, a) for a in args]
    result = await asyncio.gather(*futures)
    return {"result": list(result)}


@app.get("/health")
def health():
    return {"status": "ok", "num_docs": len(state.docids) if state else 0}


def main():
    global state, search_pool
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/paddlejob/workspace/xzj/dataset/browsecomp-plus-processed")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pool_workers", type=int, default=8,
                        help="Number of process pool workers for parallel BM25 search")
    parser.add_argument("--index_cache", default="/tmp/browsecomp_bm25_cache.pkl",
                        help="Path to cache the BM25 index (speeds up subsequent starts)")
    args = parser.parse_args()

    # Build/load index in main process (also used for health check)
    state = RetrieverState(args.data_dir, index_cache=args.index_cache)

    # Create process pool — each worker loads the index from cache independently
    print(f"Starting process pool with {args.pool_workers} workers ...", flush=True)
    search_pool = ProcessPoolExecutor(
        max_workers=args.pool_workers,
        initializer=_pool_init,
        initargs=(args.index_cache,),
    )
    # Warm up: force all workers to initialize now
    list(search_pool.map(_pool_search, [("warmup", 1)] * args.pool_workers))
    print(f"Process pool ready ({args.pool_workers} workers).", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
