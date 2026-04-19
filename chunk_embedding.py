import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from chunking import load_documents, chunk_documents

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME      = "all-MiniLM-L6-v2"   # 384-dim, fast & accurate
BATCH_SIZE      = 512
SAMPLE_SIZE     = 50_000               # set to None to embed everything
EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE     = "chunks.npy"


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Encode all chunk texts using SentenceTransformer.
    Returns a float32 numpy array of shape (n_chunks, embedding_dim).
    Auto-uses GPU if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(MODEL_NAME, device=device)

    print(f"[Embedding] Model   : {MODEL_NAME}")
    print(f"[Embedding] Device  : {device.upper()}")
    print(f"[Embedding] Dim     : {model.get_sentence_embedding_dimension()}")
    print(f"[Embedding] Chunks  : {len(chunks):,}")
    print(f"[Embedding] Batch   : {BATCH_SIZE}")
    print(f"[Embedding] Encoding ...\n")

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
    )

    return embeddings.astype(np.float32)


if __name__ == "__main__":
    # ── Load & chunk ─────────────────────────────────────────────────────────
    df     = load_documents("webmd.csv")
    chunks = chunk_documents(df)

    # ── Optional sample (for CPU environments) ────────────────────────────────
    if SAMPLE_SIZE and len(chunks) > SAMPLE_SIZE:
        import random
        random.seed(42)
        chunks = random.sample(chunks, SAMPLE_SIZE)
        print(f"[Embedding] Sampled {SAMPLE_SIZE:,} chunks for embedding\n")

    # ── Embed ─────────────────────────────────────────────────────────────────
    embeddings = embed_chunks(chunks)

    # ── Save to disk ──────────────────────────────────────────────────────────
    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(CHUNKS_FILE, np.array(chunks, dtype=object))

    print(f"\n[Embedding] Shape  : {embeddings.shape}")
    print(f"[Embedding] Saved  → {EMBEDDINGS_FILE}  ({embeddings.nbytes / 1e6:.1f} MB)")
    print(f"[Embedding] Saved  → {CHUNKS_FILE}")
    print("[Embedding] ✓ Step complete. Ready for indexing / retrieval.")
