import os
import time
import requests
from dotenv import load_dotenv
from augmentation import Augmentor

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True, encoding="utf-8")

# Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_NAME         = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
API_URL            = "https://openrouter.ai/api/v1/chat/completions"


class Generator:
    def __init__(self, top_k: int = 5):
        print(f"[Generation] Using OpenRouter model: {MODEL_NAME}")
        self.augmentor = Augmentor(top_k=top_k)
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        print("[Generation] Ready.\n")

    def _fallback_answer(self, chunks: list[dict]) -> str:
        lines = ["LLM unavailable - showing retrieved patient reviews:\n"]
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"[{i}] Drug: {c['drug']} | Condition: {c['condition']} "
                f"| Effectiveness: {c['effectiveness']}/5\n{c['text'][:300]}\n"
            )
        return "\n".join(lines)

    def generate(self, query: str) -> dict:
        aug    = self.augmentor.augment(query)
        prompt = aug["prompt"]

        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_api_key_here":
            return {
                "query"  : query,
                "answer" : self._fallback_answer(aug["chunks"]),
                "chunks" : aug["chunks"],
                "prompt" : prompt,
            }

        payload = {
            "model"     : MODEL_NAME,
            "messages"  : [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }

        try:
            for attempt in range(5):
                response = requests.post(API_URL, headers=self.headers, json=payload, timeout=60)
                if response.status_code == 429:
                    wait = 2 ** attempt
                    print(f"[Generation] Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                break
            else:
                raise RuntimeError("Rate limit exceeded after 5 retries.")
            answer = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Generation] LLM error: {e} - falling back to retrieved chunks.")
            answer = self._fallback_answer(aug["chunks"])

        return {
            "query"  : query,
            "answer" : answer,
            "chunks" : aug["chunks"],
            "prompt" : prompt,
        }

    def pretty_print(self, result: dict):
        print(f"Q: {result['query']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources ({len(result['chunks'])} reviews):")
        for i, c in enumerate(result["chunks"], 1):
            print(f"  [{i}] {c['drug']} | {c['condition']} | score={c['score']:.3f}")
        print()


if __name__ == "__main__":
    rag = Generator(top_k=5)
    queries = [
        "Does metformin help with diabetes?",
        "What are the side effects of ibuprofen?",
        "I have severe anxiety, what medication works best?",
    ]
    for query in queries:
        result = rag.generate(query)
        rag.pretty_print(result)
        print("=" * 60)
    print("[Generation] done.")

