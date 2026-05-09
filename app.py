import gradio as gr
import os
import numpy as np

# ── Indexing Pipeline ─────────────────────────────────────────────────────────

def run_indexing_pipeline(csv_path: str, chunk_size: int, chunk_overlap: int, sample_size: int):
    logs = []

    def log(msg):
        logs.append(msg)
        return "\n".join(logs)

    try:
        yield log("📄 [1/4] Loading documents...")
        from document_loading import load_documents
        df = load_documents(csv_path)
        yield log(f"✅ Loaded {len(df):,} rows")

        yield log("✂️  [2/4] Chunking documents...")
        import chunking as ck
        ck.CHUNK_SIZE    = chunk_size
        ck.CHUNK_OVERLAP = chunk_overlap
        chunks = ck.chunk_documents(df)
        yield log(f"✅ Produced {len(chunks):,} chunks")

        if sample_size > 0 and len(chunks) > sample_size:
            import random; random.seed(42)
            chunks = random.sample(chunks, int(sample_size))
        yield log(f"🔢 [3/4] Embedding {len(chunks):,} chunks (this may take a while)...")
        from chunk_embedding import embed_chunks
        embeddings = embed_chunks(chunks)
        yield log(f"✅ Embeddings shape: {embeddings.shape}")

        yield log("💾 [4/4] Building FAISS index & saving vector store...")
        import store_embeddings as se
        np.save(se.EMBEDDINGS_FILE, embeddings)
        np.save(se.CHUNKS_FILE, np.array(chunks, dtype=object))
        index = se.build_faiss_index(embeddings)
        se.save_store(index, chunks)
        yield log(f"✅ Vector store saved → {se.FAISS_INDEX_FILE}, {se.CHUNKS_STORE_FILE}")
        yield log("🎉 Indexing complete! Switch to the RAG Query tab to ask questions.")

    except Exception as e:
        yield log(f"❌ Error: {e}")


# ── RAG Query Pipeline ────────────────────────────────────────────────────────

_rag = None

def run_rag_query(query: str, top_k: int):
    if not query.strip():
        yield "⚠️ Please enter a question.", "", ""
        return

    if not (os.path.exists("faiss_index.bin") and os.path.exists("chunks_store.pkl")):
        yield "❌ Vector store not found. Run the Indexing Pipeline first.", "", ""
        return

    global _rag
    try:
        yield "⏳ Loading RAG pipeline...", "", ""
        if _rag is None:
            from generation import Generator
            _rag = Generator(top_k=top_k)

        yield "🔍 Retrieving relevant chunks & generating answer...", "", ""
        result = _rag.generate(query, top_k=top_k)

        sources_md = "\n\n".join(
            f"[{i}] {c['drug']} | {c['condition']} | score={c['score']:.3f}\n{c['text'][:200]}..."
            for i, c in enumerate(result["chunks"], 1)
        )

        yield result["answer"], sources_md, result["prompt"]

    except Exception as e:
        yield f"❌ Error: {e}", "", ""


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Medical RAG System") as demo:
    gr.Markdown("# 🏥 Medical RAG System\nBuilt on WebMD patient reviews using FAISS + Sentence Transformers")

    with gr.Tabs():

        # ── Indexing Pipeline Tab ─────────────────────────────────────────
        with gr.Tab("⚙️ Indexing Pipeline"):
            gr.Markdown("### Document Loading → Chunking → Chunk Embedding → Vector Store\nRun this once to build the vector store from the CSV.")

            with gr.Row():
                csv_input     = gr.Textbox(value="webmd.csv", label="CSV File Path")
                chunk_size    = gr.Slider(100, 1000, value=500, step=50,  label="Chunk Size (chars)")
                chunk_overlap = gr.Slider(0,   200,  value=50,  step=10,  label="Chunk Overlap (chars)")
                sample_size   = gr.Number(value=50000, label="Sample Size (0 = all)", precision=0)

            index_btn = gr.Button("▶ Run Indexing Pipeline", variant="primary")
            index_log = gr.Textbox(label="Indexing Log", lines=12, interactive=False)

            index_btn.click(
                fn=run_indexing_pipeline,
                inputs=[csv_input, chunk_size, chunk_overlap, sample_size],
                outputs=[index_log],
            )

        # ── RAG Query Pipeline Tab ────────────────────────────────────────
        with gr.Tab("🔎 RAG Query Pipeline"):
            gr.Markdown("### Retrieval → Augmentation → Generation\nAsk a medical question and get an answer grounded in patient reviews.")

            with gr.Row():
                query_input  = gr.Textbox(
                    placeholder="e.g. What are the side effects of ibuprofen?",
                    label="Your Question",
                    scale=4,
                )
                top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Top-K Retrieved Chunks", scale=1)

            ask_btn    = gr.Button("🚀 Generate Answer", variant="primary")
            answer_box = gr.Textbox(label="Generated Answer", lines=5, interactive=False)

            with gr.Accordion("📋 Retrieved Chunks", open=False):
                sources_box = gr.Textbox(label="Retrieved Sources", lines=10, interactive=False)

            with gr.Accordion("� Augmented Prompt", open=False):
                prompt_box = gr.Textbox(label="Prompt sent to LLM", lines=20, interactive=False)

            ask_btn.click(
                fn=run_rag_query,
                inputs=[query_input, top_k_slider],
                outputs=[answer_box, sources_box, prompt_box],
            )

            gr.Examples(
                examples=[
                    ["What are the side effects of ibuprofen?", 5],
                    ["Does metformin help with diabetes?", 5],
                    ["I have severe anxiety, what medication works best?", 7],
                ],
                inputs=[query_input, top_k_slider],
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
