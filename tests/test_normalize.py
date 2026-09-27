import pandas as pd
from src.normalize import (
    normalize_text,
    normalize_name,
    normalize_address,
    extract_postcode,
    extract_house_no,
    normalize_df,
)


def test_sharma_traders():
    """Sharma Traders Pvt. Ltd. -> nosuffix sharma traders."""
    clean, nosuffix = normalize_name("Sharma Traders Pvt. Ltd.")
    assert nosuffix == "sharma traders"


def test_smith_and_sons():
    """Smith & Sons Corp -> smith and sons."""
    clean, nosuffix = normalize_name("Smith & Sons Corp")
    assert nosuffix == "smith and sons"


def test_societe_generale():
    """Société Générale SARL -> societe generale."""
    clean, nosuffix = normalize_name("Société Générale SARL")
    assert nosuffix == "societe generale"


def test_bengaluru_address():
    """12 MG Rd, Near SBI ATM, Bengaluru 560 001 -> postcode 560001, house 12, address contains road."""
    addr = "12 MG Rd, Near SBI ATM, Bengaluru 560 001"
    postcode = extract_postcode(addr)
    house = extract_house_no(addr)
    norm_addr = normalize_address(addr)

    assert postcode == "560001"
    assert house == "12"
    assert "road" in norm_addr


def test_us_address():
    """500 Main St, Springfield, IL 62701-1234 -> postcode 62701."""
    addr = "500 Main St, Springfield, IL 62701-1234"
    postcode = extract_postcode(addr)
    house = extract_house_no(addr)
    norm_addr = normalize_address(addr)

    assert postcode == "62701"
    assert house == "500"
    assert "street" in norm_addr


def test_french_address():
    """15 av. des Champs-Élysées, 75008 Paris -> postcode 75008, address contains avenue and champs elysees."""
    addr = "15 av. des Champs-Élysées, 75008 Paris"
    postcode = extract_postcode(addr)
    house = extract_house_no(addr)
    norm_addr = normalize_address(addr)

    assert postcode == "75008"
    assert house == "15"
    assert "avenue" in norm_addr
    assert "champs elysees" in norm_addr


def test_normalize_df():
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["Sharma Traders Pvt. Ltd.", "Smith & Sons Corp"],
        "business_address": [
            "12 MG Rd, Bengaluru 560 001",
            "500 Main St, Springfield, IL 62701-1234",
        ],
        "country": ["India", "US"],
    })

    res = normalize_df(df)
    assert list(res.columns) == [
        "entity_id",
        "country",
        "name_clean",
        "name_nosuffix",
        "addr_clean",
        "postcode",
        "house_no",
    ]
    assert res.loc[0, "name_nosuffix"] == "sharma traders"
    assert res.loc[0, "postcode"] == "560001"
    assert res.loc[1, "name_nosuffix"] == "smith and sons"
    assert res.loc[1, "postcode"] == "62701"
