import pandas as pd
from document_loading import load_documents

# ── Config ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks


def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple recursive character-level splitter.
    Tries to split on paragraphs → sentences → words → characters.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # last resort: hard split by characters
        if not seps or seps[0] == "":
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        sep = seps[0]
        parts = text.split(sep)
        chunks, current = [], ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # if a single part is still too long, recurse with next separator
                if len(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    raw_chunks = _split(text, separators)

    # Apply overlap: each chunk starts `overlap` chars before the previous ended
    if overlap == 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = overlapped[-1][-overlap:]
        overlapped.append(prev_tail + " " + raw_chunks[i])

    return overlapped


def chunk_documents(df: pd.DataFrame) -> list[dict]:
    """
    Takes the loaded DataFrame and returns a list of chunk dicts.
    Each dict contains the chunk text + metadata from the original row.
    Uses vectorized apply for speed.
    """
    df = df.dropna(subset=["Reviews"]).reset_index(drop=True)
    print(f"[Chunking] {len(df):,} rows with valid Reviews")

    # Build metadata records as a list of dicts (fast)
    meta_cols = ["Drug", "Condition", "Age", "Sex", "Effectiveness", "Satisfaction", "EaseofUse", "Date"]
    records   = df[meta_cols].to_dict("records")
    reviews   = df["Reviews"].astype(str).tolist()

    chunks = []
    for idx, (text, meta) in enumerate(zip(reviews, records)):
        text = text.strip()
        if not text:
            continue
        parts = recursive_split(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(parts):
            chunks.append({
                "chunk_id"     : f"{idx}_{i}",
                "text"         : chunk,
                "drug"         : meta["Drug"],
                "condition"    : meta["Condition"],
                "age"          : meta["Age"],
                "sex"          : meta["Sex"],
                "effectiveness": meta["Effectiveness"],
                "satisfaction" : meta["Satisfaction"],
                "ease_of_use"  : meta["EaseofUse"],
                "date"         : meta["Date"],
            })

    return chunks


if __name__ == "__main__":
    df     = load_documents("webmd.csv")
    chunks = chunk_documents(df)

    print(f"\n[Chunking] Total chunks produced : {len(chunks):,}")
    print(f"[Chunking] Avg chars per chunk   : {sum(len(c['text']) for c in chunks) // len(chunks)}")

    print("\n── Sample chunk ──")
    sample = chunks[5]
    for k, v in sample.items():
        print(f"  {k:15}: {str(v)[:120]}")

    print("\n[Chunking] ✓ Step complete. Chunks ready for embedding.")
