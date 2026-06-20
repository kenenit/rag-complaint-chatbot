"""
preprocess.py — Text cleaning and filtering for CFPB complaint narratives.
"""

import re
import pandas as pd


# Products we care about (mapped to standard names)
TARGET_PRODUCTS = {
    "Credit card": "Credit Card",
    "Credit Card": "Credit Card",
    "Payday loan": "Personal Loan",
    "Personal loan": "Personal Loan",
    "Personal Loan": "Personal Loan",
    "Checking or savings account": "Savings Account",
    "Savings account": "Savings Account",
    "Money transfer, virtual currency, or service": "Money Transfer",
    "Money transfers": "Money Transfer",
}

# Boilerplate phrases commonly found in CFPB narratives
BOILERPLATE_PATTERNS = [
    r"i am writing to (file|submit) (a )?complaint",
    r"i am filing this complaint",
    r"xxxx",          # redacted account numbers
    r"\bxx\b",        # redacted dates/numbers
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CFPB CSV dataset."""
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df):,} records with columns: {list(df.columns)}")
    return df


def filter_products(df: pd.DataFrame, product_col: str = "Product") -> pd.DataFrame:
    """Keep only the four target product categories."""
    df = df.copy()
    df["product_category"] = df[product_col].map(TARGET_PRODUCTS)
    filtered = df[df["product_category"].notna()].reset_index(drop=True)
    print(f"After product filter: {len(filtered):,} records")
    return filtered


def filter_narratives(df: pd.DataFrame, narrative_col: str = "Consumer complaint narrative") -> pd.DataFrame:
    """Remove records with missing narratives."""
    df = df.copy()
    df = df[df[narrative_col].notna() & (df[narrative_col].str.strip() != "")]
    print(f"After narrative filter: {len(df):,} records")
    return df.reset_index(drop=True)


def clean_text(text: str) -> str:
    """
    Clean a single complaint narrative:
    - Lowercase
    - Remove boilerplate phrases
    - Remove special characters (keep alphanumeric + basic punctuation)
    - Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Remove special characters, keep sentence structure
    text = re.sub(r"[^a-z0-9\s.,!?;:'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_pipeline(filepath: str, narrative_col: str = "Consumer complaint narrative") -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Load raw data
    2. Filter to target products
    3. Remove empty narratives
    4. Clean narrative text
    5. Return cleaned DataFrame
    """
    df = load_data(filepath)
    df = filter_products(df)
    df = filter_narratives(df, narrative_col)
    df["cleaned_narrative"] = df[narrative_col].apply(clean_text)
    # Drop rows where cleaning resulted in very short text
    df = df[df["cleaned_narrative"].str.len() > 50].reset_index(drop=True)
    print(f"Final cleaned dataset: {len(df):,} records")
    return df
