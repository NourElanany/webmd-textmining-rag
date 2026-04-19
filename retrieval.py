import numpy as np
from sentence_transformers import SentenceTransformer
from store_embeddings import load_store, search

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"   # must match embedding step
TOP_K      = 5


class Retriever:
    def __init__(self, top_k: int = TOP_K):
        print(f"[Retrieval] Loading model  : {MODEL_NAME}")
        self.model  = SentenceTransformer(MODEL_NAME)
        self.index, self.chunks = load_store()
        self.top_k  = top_k
        print(f"[Retrieval] Ready. Index has {self.index.ntotal:,} vectors.\n")

    def embed_query(self, query: str) -> np.ndarray:
        """Encode and L2-normalize the query string."""
        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

    def retrieve(self, query: str) -> list[dict]:
        """Return top-k most relevant chunks for the given query."""
        query_vec = self.embed_query(query)
        results   = search(query_vec, self.index, self.chunks, top_k=self.top_k)
        return results

    def pretty_print(self, query: str, results: list[dict]):
        print(f"Query : \"{query}\"")
        print("─" * 60)
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r['score']:.4f} | {r['drug']} | {r['condition']}")
            print(f"     {r['text'][:200]}")
            print()


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = Retriever(top_k=TOP_K)

    test_queries = [
        "What are the side effects of ibuprofen?",
        "Does metformin help with diabetes?",
        "I have severe anxiety, what medication works best?",
    ]

    for query in test_queries:
        results = retriever.retrieve(query)
        retriever.pretty_print(query, results)
        print("=" * 60)
        print()
