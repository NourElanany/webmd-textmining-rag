import numpy as np
import faiss
import pickle

# ── Config ───────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE     = "chunks.npy"
FAISS_INDEX_FILE = "faiss_index.bin"
CHUNKS_STORE_FILE = "chunks_store.pkl"


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP (Inner Product) index.
    Since embeddings are L2-normalized, inner product == cosine similarity.
    """
    dim = embeddings.shape[1]

    # IndexFlatIP = exact search, cosine similarity (embeddings are normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[Store] Index type : IndexFlatIP (exact cosine search)")
    print(f"[Store] Dimensions : {dim}")
    print(f"[Store] Vectors    : {index.ntotal:,}")
    return index


def save_store(index: faiss.Index, chunks: list[dict]):
    """Persist the FAISS index and chunk metadata to disk."""
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(CHUNKS_STORE_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[Store] Saved → {FAISS_INDEX_FILE}")
    print(f"[Store] Saved → {CHUNKS_STORE_FILE}")


def load_store() -> tuple[faiss.Index, list[dict]]:
    """Load the FAISS index and chunk metadata from disk."""
    index  = faiss.read_index(FAISS_INDEX_FILE)
    with open(CHUNKS_STORE_FILE, "rb") as f:
        chunks = pickle.load(f)
    print(f"[Store] Loaded index  : {index.ntotal:,} vectors")
    print(f"[Store] Loaded chunks : {len(chunks):,}")
    return index, chunks


def search(query_embedding: np.ndarray, index: faiss.Index,
           chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Retrieve top-k most similar chunks for a query embedding.
    Returns list of dicts with chunk data + similarity score.
    """
    query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        result = dict(chunks[idx])
        result["score"] = float(score)
        results.append(result)
    return results


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load embeddings & chunks produced by chunk_embedding.py
    print("[Store] Loading embeddings ...")
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    chunks     = list(np.load(CHUNKS_FILE, allow_pickle=True))
    print(f"[Store] Embeddings shape : {embeddings.shape}")
    print(f"[Store] Chunks loaded    : {len(chunks):,}\n")

    # Build & save
    index = build_faiss_index(embeddings)
    save_store(index, chunks)

    # Verify: reload and run a sample search
    print("\n[Store] Verifying with sample search ...")
    index, chunks = load_store()

    # Use the first embedding as a dummy query
    sample_query = embeddings[0]
    results = search(sample_query, index, chunks, top_k=3)

    print("\n── Top-3 results for sample query ──")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] score={r['score']:.4f}")
        print(f"       drug      : {r['drug']}")
        print(f"       condition : {r['condition']}")
        print(f"       text      : {r['text'][:120]}...")

    print("\n[Store] ✓ Step complete. Vector store ready for retrieval.")
