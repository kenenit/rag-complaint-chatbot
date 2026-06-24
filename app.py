import re, os, warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = 'vector_store/chroma'
EMBED_MODEL      = 'sentence-transformers/all-MiniLM-L6-v2'
GROQ_API_KEY     = "YOUR_GROQ_KEY_HERE"
TOP_K            = 5

# ── Load models ───────────────────────────────────────────────────────
print('Loading embedding model...')
embed_model = SentenceTransformer(EMBED_MODEL)

print('Loading ChromaDB...')
client     = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
collection = client.get_collection('complaints')
print(f'Loaded {collection.count():,} chunks')

groq_client = Groq(api_key=GROQ_API_KEY)

# ── Prompt template ───────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial.
Your task is to answer questions about customer complaints.
Use ONLY the following retrieved complaint excerpts to formulate your answer.
If the context does not contain enough information, clearly state that.
Be concise, factual, and cite patterns you observe across multiple complaints.

Context:
{context}

Question: {question}

Answer:"""

# ── RAG functions ─────────────────────────────────────────────────────
def retrieve(query, k=TOP_K, product_filter=None):
    query_vec = embed_model.encode([query], normalize_embeddings=True).tolist()
    where     = {'product_category': product_filter} if product_filter else None
    results   = collection.query(
        query_embeddings=query_vec,
        n_results=k,
        where=where,
        include=['documents', 'metadatas', 'distances']
    )
    chunks = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        chunks.append({'text': doc, 'metadata': meta, 'score': round(1 - dist, 4)})
    return chunks

def build_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk['metadata']
        parts.append(
            f"[{i}] Product: {meta.get('product_category','?')} | "
            f"Issue: {meta.get('issue','?')}\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)

def generate(question, chunks):
    context  = build_context(chunks)
    prompt   = PROMPT_TEMPLATE.format(context=context, question=question)
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content

def format_sources(chunks):
    lines = []
    for i, chunk in enumerate(chunks, 1):
        meta  = chunk['metadata']
        score = chunk['score']
        prod  = meta.get('product_category', 'N/A')
        issue = meta.get('issue', 'N/A')
        text  = chunk['text'][:300] + ('...' if len(chunk['text']) > 300 else '')
        lines.append(
            f"**Source {i}** — {prod} | {issue} _(similarity: {score})_\n\n> {text}"
        )
    return "\n\n---\n\n".join(lines)

# ── Gradio handler ────────────────────────────────────────────────────
def answer_question(question, product_filter):
    if not question.strip():
        return "Please enter a question.", ""
    filter_val = None if product_filter == "All Products" else product_filter
    try:
        chunks  = retrieve(question, product_filter=filter_val)
        answer  = generate(question, chunks)
        sources = format_sources(chunks)
        return answer, sources
    except Exception as e:
        return f"Error: {str(e)}", ""

# ── Gradio UI ─────────────────────────────────────────────────────────
PRODUCTS = ["All Products", "Credit Card", "Personal Loan",
            "Savings Account", "Money Transfer"]

with gr.Blocks(title="CrediTrust Complaint Chatbot", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🏦 CrediTrust Complaint Intelligence Chatbot
    Ask plain-English questions about customer complaints across all product lines.
    The system retrieves relevant complaint narratives and generates a grounded answer.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder='e.g. "Why are customers unhappy with credit cards?"',
                lines=2
            )
        with gr.Column(scale=1):
            product_dd = gr.Dropdown(
                choices=PRODUCTS,
                value="All Products",
                label="Filter by Product"
            )

    with gr.Row():
        submit_btn = gr.Button("🔍 Ask", variant="primary")
        clear_btn  = gr.Button("✕ Clear")

    answer_box  = gr.Textbox(label="Generated Answer", lines=8, interactive=False)
    sources_box = gr.Markdown(label="Retrieved Sources")

    gr.Examples(
        examples=[
            ["Why are customers unhappy with credit cards?", "All Products"],
            ["What fraud issues do credit card customers face?", "Credit Card"],
            ["What are the most common savings account complaints?", "Savings Account"],
            ["What should product teams prioritise to reduce complaints?", "All Products"],
        ],
        inputs=[question_box, product_dd]
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[question_box, product_dd],
        outputs=[answer_box, sources_box]
    )
    clear_btn.click(
        fn=lambda: ("", "All Products", "", ""),
        outputs=[question_box, product_dd, answer_box, sources_box]
    )

if __name__ == "__main__":
    demo.launch(share=False)
