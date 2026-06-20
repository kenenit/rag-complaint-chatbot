"""Unit tests for src/chunker.py"""

import pytest
from src.chunker import create_splitter, chunk_dataframe
import pandas as pd


def test_splitter_chunk_size():
    splitter = create_splitter(chunk_size=100, chunk_overlap=10)
    text = "a " * 200  # 400 chars
    chunks = splitter.split_text(text)
    assert all(len(c) <= 110 for c in chunks)  # allow slight overflow at word boundary


def test_chunk_dataframe_returns_list():
    df = pd.DataFrame({
        "cleaned_narrative": ["This is a test complaint narrative. " * 10],
        "product_category": ["Credit Card"],
        "Complaint ID": [1],
        "Product": ["Credit Card"],
        "Issue": ["Billing"],
        "Sub-issue": [""],
        "Company": ["TestBank"],
        "State": ["CA"],
        "Date received": ["2024-01-01"],
    })
    chunks = chunk_dataframe(df, chunk_size=100, chunk_overlap=10)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert "text" in chunks[0]
    assert "product_category" in chunks[0]
