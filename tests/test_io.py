from src import io


def test_write_read_round_trip(tmp_path):
    path = str(tmp_path / "out" / "m.tsv")
    io.write_id_lists({"S1-00001": ["S2-00047", "S3-00812", "S2-00047"]},
                      ["S1-00001", "S1-00002"], path, "matched_entity_ids")

    text = open(path, newline="").read()
    assert text.splitlines() == ["source1_entity_id\tmatched_entity_ids",
                                 "S1-00001\tS2-00047,S3-00812", "S1-00002\t"]
    assert '"' not in text and "\r" not in text

    df = io.read_tsv(path)
    assert list(df.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert df.loc[1, "matched_entity_ids"] == ""          # empty stays empty, not NaN
    assert df.loc[0, "matched_entity_ids"].startswith("S2-00047")  # IDs stay strings


def test_ground_truth_empty_set(tmp_path, monkeypatch):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-00047, S3-00812\nS1-2\t\n")
    monkeypatch.setitem(io.PATHS, "dataset", str(tmp_path))
    assert io.load_ground_truth() == {"S1-1": {"S2-00047", "S3-00812"}, "S1-2": set()}


def test_parquet_round_trip(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"s1_id": ["S1-00001"], "cand_id": ["S2-00047"], "blockers": ["name|postcode"]})
    path = str(tmp_path / "c.parquet")
    io.write_parquet(df, path)
    assert io.read_parquet(path).equals(df)
