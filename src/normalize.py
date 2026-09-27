"""Text normalization (owner: Lavanya, task L4). Country-agnostic cleaning."""
import re
import unicodedata

import pandas as pd

LEGAL_SUFFIXES = {
    "pvt",
    "private",
    "ltd",
    "limited",
    "llp",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
    "gmbh",
    "sarl",
    "sas",
    "sa",
    "eurl",
    "sasu",
    "sci",
}

ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "pl": "place",
    "ct": "court",
    "ter": "terrace",
    "twp": "township",
    "mt": "mount",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "nr": "near",
    "opp": "opposite",
    "mg": "mahatma gandhi",
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode NFKD and strip combining marks."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def normalize_text(s: str) -> str:
    """Lowercase, strip accents, & -> and, punctuation -> space, collapse spaces."""
    if pd.isna(s) or s is None:
        return ""

    text = str(s).strip().lower()
    text = normalize_unicode(text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


clean_text = normalize_text  # alias


def normalize_abbreviations(text: str) -> str:
    words = text.split()
    words = [ADDRESS_ABBREVIATIONS.get(w, w) for w in words]
    return " ".join(words)


def remove_legal_suffix(text: str) -> str:
    words = text.split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def normalize_name(s: str) -> tuple[str, str]:
    """Returns (name_clean, name_nosuffix)."""
    clean = normalize_text(s)
    nosuffix = remove_legal_suffix(clean)
    return clean, nosuffix


def normalize_address(s: str) -> str:
    """Normalize and expand address abbreviations."""
    clean = normalize_text(s)
    return normalize_abbreviations(clean)


def extract_postcode(raw_address: str) -> str:
    """Extract postcode from raw address.
    
    Handles:
    - 6-digit Indian PIN (e.g. 560 001 -> 560001 or 110001)
    - 5-digit US / French codes (e.g. 75008)
    - US ZIP+4 trimmed to 5 digits (e.g. 62701-1234 -> 62701)
    Takes the last match.
    """
    if pd.isna(raw_address) or not raw_address:
        return ""

    raw_str = str(raw_address).strip()

    # Match Indian PIN with space, e.g. 560 001
    pin_spaced = re.findall(r"\b\d{3}\s+\d{3}\b", raw_str)
    # Match standard 5 or 6 digit codes, or ZIP+4
    std_codes = re.findall(r"\b(\d{5})(?:-\d{4})?\b|\b(\d{6})\b", raw_str)

    # Check last occurrence in string
    matches = list(re.finditer(r"\b(\d{3}\s+\d{3})\b|\b(\d{5})(?:-\d{4})?\b|\b(\d{6})\b", raw_str))
    if not matches:
        return ""

    last_match = matches[-1]
    g1, g2, g3 = last_match.groups()
    if g1:
        return re.sub(r"\s+", "", g1)
    if g2:
        return g2
    if g3:
        return g3
    return ""


def extract_house_no(raw_address: str) -> str:
    """Extract first token that starts with a digit and is not the postcode."""
    if pd.isna(raw_address) or not raw_address:
        return ""

    postcode = extract_postcode(raw_address)
    tokens = re.findall(r"\b\d+[A-Za-z]?(?:-\d+[A-Za-z]?)?\b", str(raw_address))
    for t in tokens:
        clean_t = re.sub(r"\s+", "", t)
        if clean_t and clean_t != postcode:
            return t
    return ""


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe to contract columns of TRD §3.2."""
    result = pd.DataFrame()
    result["entity_id"] = df["entity_id"].astype(str)
    result["country"] = df["country"].fillna("").astype(str)

    name_clean = []
    name_nosuffix = []
    for val in df["business_name"]:
        c, ns = normalize_name(val)
        name_clean.append(c)
        name_nosuffix.append(ns)

    result["name_clean"] = name_clean
    result["name_nosuffix"] = name_nosuffix
    result["addr_clean"] = df["business_address"].apply(normalize_address)
    result["postcode"] = df["business_address"].apply(extract_postcode)
    result["house_no"] = df["business_address"].apply(extract_house_no)

    return result.fillna("")


normalize_dataframe = normalize_df  # alias for backwards compatibility
