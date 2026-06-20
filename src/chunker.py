"""
chunker.py — Text chunking strategies for complaint narratives.
"""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd


def create_splitter(chunk_size: int = 500, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    """Create a LangChain RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_dataframe(
    df: pd.DataFrame,
    text_col: str = "cleaned_narrative",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunk all narratives in a DataFrame.
    Returns a list of dicts, each representing one chunk with metadata.
    """
    splitter = create_splitter(chunk_size, chunk_overlap)
    chunks = []

    for _, row in df.iterrows():
        text = row[text_col]
        if not text:
            continue
        split_texts = splitter.split_text(text)
        for idx, chunk_text in enumerate(split_texts):
            chunks.append({
                "text": chunk_text,
                "complaint_id": row.get("Complaint ID", row.name),
                "product_category": row.get("product_category", ""),
                "product": row.get("Product", ""),
                "issue": row.get("Issue", ""),
                "sub_issue": row.get("Sub-issue", ""),
                "company": row.get("Company", ""),
                "state": row.get("State", ""),
                "date_received": str(row.get("Date received", "")),
                "chunk_index": idx,
                "total_chunks": len(split_texts),
            })

    print(f"Created {len(chunks):,} chunks from {len(df):,} complaints")
    return chunks
