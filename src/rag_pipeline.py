"""
rag_pipeline.py — End-to-end RAG orchestration (retriever + generator).
"""

from typing import List, Dict, Any, Optional
from src.retriever import ChromaRetriever, FAISSRetriever
from src.generator import LLMGenerator, build_prompt


class RAGPipeline:
    """
    Orchestrates the full RAG flow:
    1. Retrieve relevant chunks via semantic search
    2. Build a grounded prompt
    3. Generate an answer with the LLM
    """

    def __init__(
        self,
        retriever_type: str = "chroma",  # "chroma" or "faiss"
        vector_store_dir: str = "vector_store",
        llm_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        k: int = 5,
        device: str = "cpu",
    ):
        # Load retriever
        if retriever_type == "chroma":
            self.retriever = ChromaRetriever(persist_dir=f"{vector_store_dir}/chroma")
        elif retriever_type == "faiss":
            self.retriever = FAISSRetriever(persist_dir=f"{vector_store_dir}/faiss")
        else:
            raise ValueError(f"Unknown retriever_type: {retriever_type}")

        self.generator = LLMGenerator(model_id=llm_model_id, device=device)
        self.k = k

    def ask(
        self,
        question: str,
        product_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ask a question and return the generated answer plus source chunks.

        Returns:
            {
                "question": str,
                "answer": str,
                "sources": List[Dict]   # retrieved chunks with metadata
            }
        """
        chunks = self.retriever.retrieve(question, k=self.k, product_filter=product_filter)
        prompt = build_prompt(question, chunks)
        answer = self.generator.generate(prompt)
        return {
            "question": question,
            "answer": answer,
            "sources": chunks,
        }
