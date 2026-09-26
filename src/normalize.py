"""Text normalization (owner: Lavanya, task L4). Placeholder from S0."""
import pandas as pd


def normalize_text(s: str) -> str:
    raise NotImplementedError("L4 (Lavanya)")


def normalize_name(s: str) -> tuple[str, str]:
    raise NotImplementedError("L4 (Lavanya)")


def normalize_address(s: str) -> str:
    raise NotImplementedError("L4 (Lavanya)")


def extract_postcode(raw_address: str) -> str:
    raise NotImplementedError("L4 (Lavanya)")


def extract_house_no(raw_address: str) -> str:
    raise NotImplementedError("L4 (Lavanya)")


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("L4 (Lavanya)")
