"""Unit tests for src/preprocess.py"""

import pytest
from src.preprocess import clean_text, filter_products
import pandas as pd


def test_clean_text_lowercase():
    assert clean_text("HELLO WORLD") == "hello world"


def test_clean_text_removes_boilerplate():
    text = "I am writing to file a complaint about my credit card."
    result = clean_text(text)
    assert "i am writing to" not in result


def test_clean_text_removes_special_chars():
    result = clean_text("Hello @#$% world!!!")
    assert "@" not in result
    assert "#" not in result


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_filter_products_keeps_target():
    df = pd.DataFrame({"Product": ["Credit card", "Mortgage", "Personal loan", "Auto loan"]})
    filtered = filter_products(df)
    assert set(filtered["product_category"].unique()).issubset(
        {"Credit Card", "Personal Loan", "Savings Account", "Money Transfer"}
    )
    assert len(filtered) == 2  # Credit card + Personal loan
