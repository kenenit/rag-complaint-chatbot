"""
generator.py — LLM-based answer generation with retrieved context.
"""

from typing import List, Dict, Any


PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial. \
Your task is to answer questions about customer complaints based on the retrieved complaint excerpts below.
Use ONLY the provided context to formulate your answer.
If the context does not contain enough information to answer the question, clearly state that.
Be concise, factual, and cite patterns you observe across multiple complaints where relevant.

Context:
{context}

Question: {question}

Answer:"""


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        product = meta.get("product_category", "Unknown")
        issue = meta.get("issue", "")
        header = f"[{i}] Product: {product} | Issue: {issue}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Fill the prompt template with question and retrieved context."""
    context = build_context(chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


class LLMGenerator:
    """Wraps a HuggingFace pipeline for text generation."""

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = "cpu"):
        from transformers import pipeline
        print(f"Loading LLM: {model_id}")
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device=device,
            max_new_tokens=512,
            do_sample=False,
        )

    def generate(self, prompt: str) -> str:
        """Run the LLM on a formatted prompt and return the generated text."""
        output = self.pipe(prompt)[0]["generated_text"]
        # Strip the prompt prefix to return only the answer
        if "Answer:" in output:
            return output.split("Answer:")[-1].strip()
        return output.strip()
