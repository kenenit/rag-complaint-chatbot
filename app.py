"""
app.py — Gradio chat interface for the CrediTrust RAG Complaint Chatbot.
Run with: python app.py
"""

import gradio as gr
from src.rag_pipeline import RAGPipeline

# ── Configuration ─────────────────────────────────────────────────────────────
PRODUCT_CHOICES = ["All Products", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]

# Initialise pipeline (loaded once at startup)
pipeline = None  # Lazy-loaded on first query to avoid blocking startup


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline(
            retriever_type="chroma",
            vector_store_dir="vector_store",
            k=5,
        )
    return pipeline


def answer_question(question: str, product_filter: str) -> tuple[str, str]:
    """Handler called by Gradio on Submit."""
    if not question.strip():
        return "Please enter a question.", ""

    filter_val = None if product_filter == "All Products" else product_filter
    result = get_pipeline().ask(question, product_filter=filter_val)

    answer = result["answer"]
    sources_md = format_sources(result["sources"])
    return answer, sources_md


def format_sources(chunks: list) -> str:
    """Format retrieved chunks as readable Markdown."""
    if not chunks:
        return "_No sources retrieved._"
    lines = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        product = meta.get("product_category", "N/A")
        issue = meta.get("issue", "N/A")
        score = chunk.get("score", 0)
        lines.append(
            f"**Source {i}** — {product} | {issue} _(score: {score:.3f})_\n\n"
            f"> {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}"
        )
    return "\n\n---\n\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="CrediTrust Complaint Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏦 CrediTrust Complaint Intelligence Chatbot
        Ask plain-English questions about customer complaints across our product lines.
        The system retrieves relevant complaint narratives and generates a synthesised answer.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder='e.g. "Why are customers unhappy with credit cards?"',
                lines=2,
            )
        with gr.Column(scale=1):
            product_dd = gr.Dropdown(
                choices=PRODUCT_CHOICES,
                value="All Products",
                label="Filter by Product",
            )

    with gr.Row():
        submit_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear")

    answer_box = gr.Textbox(label="Generated Answer", lines=6, interactive=False)
    sources_box = gr.Markdown(label="Retrieved Sources")

    submit_btn.click(
        fn=answer_question,
        inputs=[question_box, product_dd],
        outputs=[answer_box, sources_box],
    )
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[question_box, answer_box, sources_box],
    )

if __name__ == "__main__":
    demo.launch(share=False)
