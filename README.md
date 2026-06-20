# 🏦 CrediTrust Complaint Intelligence Chatbot

A Retrieval-Augmented Generation (RAG) system that transforms raw CFPB customer complaint data into actionable insights for product, support, and compliance teams at CrediTrust Financial.

## Project Structure

```
rag-complaint-chatbot/
├── .github/workflows/unittests.yml   # CI pipeline
├── data/
│   ├── raw/                          # Raw CFPB dataset (not committed)
│   └── processed/                    # Cleaned/filtered data (not committed)
├── vector_store/                     # Persisted FAISS/ChromaDB index
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_chunking_embedding.ipynb
│   └── 03_rag_evaluation.ipynb
├── src/
│   ├── preprocess.py                 # Data cleaning and filtering
│   ├── chunker.py                    # Text chunking
│   ├── embedder.py                   # Embedding generation & vector store
│   ├── retriever.py                  # Semantic search (Chroma / FAISS)
│   ├── generator.py                  # LLM answer generation
│   └── rag_pipeline.py               # End-to-end RAG orchestration
├── tests/
│   ├── test_preprocess.py
│   └── test_chunker.py
├── app.py                            # Gradio chat interface
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd rag-complaint-chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Place raw CFPB data
cp /path/to/complaints.csv data/raw/

# 3. Run EDA & preprocessing (Task 1)
jupyter lab notebooks/01_eda_preprocessing.ipynb

# 4. Build vector store (Task 2)
jupyter lab notebooks/02_chunking_embedding.ipynb

# 5. Launch the chatbot UI
python app.py
```

## Tasks

| Task | Description | Status |
|------|-------------|--------|
| 1 | EDA & Preprocessing | 🔲 |
| 2 | Chunking, Embedding & Vector Store | 🔲 |
| 3 | RAG Core Logic & Evaluation | 🔲 |
| 4 | Gradio Chat Interface | 🔲 |

## Key Design Decisions

- **Embedding model:** `all-MiniLM-L6-v2` — fast, lightweight, strong semantic performance on short texts
- **Vector DB:** ChromaDB (primary) with FAISS as alternative
- **Chunk size:** 500 chars / 50 overlap — matches the pre-built store spec
- **LLM:** Mistral-7B-Instruct-v0.2 (configurable)

## Team

- 10 Academy Week 7 Challenge — Jun 17–23, 2026
