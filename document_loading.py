import pandas as pd

# ── 1. Load the raw CSV ──────────────────────────────────────────────────────
def load_documents(filepath: str = "webmd.csv") -> pd.DataFrame:
    """
    Load the WebMD CSV and treat each row as a 'document'.
    Returns a cleaned DataFrame ready for the next pipeline step.
    """
    df = pd.read_csv(filepath)

    print(f"[Document Loading] Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"[Document Loading] Columns: {list(df.columns)}\n")

    # ── 2. Basic inspection ──────────────────────────────────────────────────
    print("── Sample (first 3 rows) ──")
    print(df.head(3).to_string())
    print()

    print("── Missing values per column ──")
    print(df.isnull().sum())
    print()

    # ── 3. Drop fully-empty rows ─────────────────────────────────────────────
    before = len(df)
    df.dropna(how="all", inplace=True)
    print(f"[Document Loading] Dropped {before - len(df)} fully-empty rows → {len(df):,} remaining\n")

    return df


if __name__ == "__main__":
    df = load_documents("webmd.csv")
    print("[Document Loading] ✓ Step complete. DataFrame is ready for next step.")
