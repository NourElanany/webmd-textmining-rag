from retrieval import Retriever

# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful medical information assistant. 
Answer the user's question using ONLY the patient reviews provided below.
Be concise, factual, and always remind the user to consult a doctor."""

PROMPT_TEMPLATE = """{system}

--- Patient Reviews (retrieved context) ---
{context}
-------------------------------------------

User Question: {question}

Answer:"""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Review {i}] Drug: {chunk['drug']} | Condition: {chunk['condition']} "
            f"| Effectiveness: {chunk['effectiveness']}/5 | Satisfaction: {chunk['satisfaction']}/5\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Combine system prompt + retrieved context + user query into final prompt."""
    context = build_context(chunks)
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=query,
    )


class Augmentor:
    def __init__(self, top_k: int = 5):
        self.retriever = Retriever(top_k=top_k)

    def augment(self, query: str) -> dict:
        """
        Retrieve relevant chunks and build the augmented prompt.
        Returns a dict with query, chunks, context, and full prompt.
        """
        chunks  = self.retriever.retrieve(query)
        context = build_context(chunks)
        prompt  = build_prompt(query, chunks)

        return {
            "query"  : query,
            "chunks" : chunks,
            "context": context,
            "prompt" : prompt,
        }


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    augmentor = Augmentor(top_k=5)

    query  = "Does metformin help with diabetes?"
    result = augmentor.augment(query)

    print(f"Query   : {result['query']}")
    print(f"Chunks  : {len(result['chunks'])} retrieved")
    print("\n── Augmented Prompt ──\n")
    print(result["prompt"])
    print("\n[Augmentation] ✓ Step complete. Prompt ready for generation.")
