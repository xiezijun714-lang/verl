"""
Offline script: encode all BrowseComp-Plus documents with Qwen3-Embedding and build FAISS index.
Run this ONCE before starting the dense retrieval server.

Usage:
    cd /root/paddlejob/workspace/xzj/verl
    source /root/paddlejob/workspace/xzj/venv_verl/bin/activate

    # Step 1 (if needed): install faiss
    pip install faiss-gpu   # recommended (GPU accelerated)
    # or: pip install faiss-cpu

    # Step 2: build index
    python examples/sglang_multiturn/build_embed_index.py \
        --data_dir /root/paddlejob/workspace/xzj/dataset/browsecomp-plus-processed \
        --model Qwen/Qwen3-Embedding \
        --device cuda:7 \
        --batch_size 256 \
        --output /tmp/browsecomp_dense_cache.pkl

Expected runtime: ~10-20 min for 67K docs on A100
Output file size: ~1 GB (67K * 4096 dim * 4 bytes)
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd


def build_corpus(data_dir: str) -> dict:
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

    print(f"  Total unique documents: {len(seen)}", flush=True)
    return seen


def extract_title(text: str, url: str = "") -> str:
    m = re.search(r"^---\s*\ntitle:\s*(.+?)\n", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    if url:
        return url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    return "Unknown"


def encode_all(texts, tokenizer, model, device, batch_size, max_length):
    import torch
    import torch.nn.functional as F

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        # Last token pooling (Qwen3-Embedding decoder arch)
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        batch_size_actual = out.last_hidden_state.shape[0]
        emb = out.last_hidden_state[
            torch.arange(batch_size_actual, device=device), seq_lens
        ]
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().float().numpy())

        done = i + len(batch)
        if done % (batch_size * 20) == 0 or done == len(texts):
            print(f"  [{done}/{len(texts)}] encoded", flush=True)

    return np.vstack(all_embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/paddlejob/workspace/xzj/dataset/browsecomp-plus-processed")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--output", default="/tmp/browsecomp_dense_cache.pkl")
    args = parser.parse_args()

    # Check faiss
    try:
        import faiss
    except ImportError:
        print("ERROR: faiss not found. Install with:", flush=True)
        print("  pip install faiss-gpu   # GPU (recommended)", flush=True)
        print("  pip install faiss-cpu   # CPU fallback", flush=True)
        sys.exit(1)

    import torch
    from transformers import AutoTokenizer, AutoModel

    # Build corpus
    doc_map = build_corpus(args.data_dir)
    docids = list(doc_map.keys())
    docs = [doc_map[d] for d in docids]

    contents = []
    raw_texts = []
    for doc in docs:
        title = extract_title(doc.get("text", ""), doc.get("url", ""))
        body = doc.get("text", "")
        contents.append(f"{title}\n{body}")
        raw_texts.append(body)

    # Load model
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Loading {args.model} on {args.device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    model = model.to(device).eval()
    print("  Model loaded.", flush=True)

    # Encode documents
    print(f"Encoding {len(raw_texts)} documents (batch_size={args.batch_size}) ...", flush=True)
    embeddings = encode_all(raw_texts, tokenizer, model, device, args.batch_size, args.max_doc_length)
    print(f"  Embeddings shape: {embeddings.shape}", flush=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (dim={dim}) ...", flush=True)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  Index built: {index.ntotal} vectors", flush=True)

    # Quick sanity check
    q = embeddings[:1]
    scores, ids = index.search(q, 3)
    print(f"  Sanity check: top-3 scores for doc[0]: {scores[0].tolist()}", flush=True)

    # Save cache
    print(f"Saving to {args.output} ...", flush=True)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({
            "docids": docids,
            "contents": contents,
            "embeddings": embeddings,
        }, f)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"  Saved {size_mb:.0f} MB to {args.output}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
