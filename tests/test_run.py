from src import io, run


def _write(path, rows):
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                    + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


def test_baseline_outputs(tmp_path, monkeypatch):
    test = tmp_path / "dataset" / "test"
    test.mkdir(parents=True)
    _write(test / "test_source1.tsv", [("S1-1", "Sharma Traders", "", "India"),
                                       ("S1-2", "Boulangerie Paul", "", "France"),
                                       ("S1-3", "", "", "US")])
    _write(test / "test_source2.tsv", [("S2-1", "SHARMA TRADERS ", "", "India"),
                                       ("S2-2", "Acme Corp", "", "US")])
    _write(test / "test_source3.tsv", [("S3-1", "Boulangerie Paul", "", "France"),
                                       ("S3-2", "Zeta Labs", "", "US")])
    monkeypatch.setitem(run.PATHS, "dataset", str(tmp_path / "dataset"))
    monkeypatch.setitem(run.PATHS, "output", str(tmp_path / "output"))

    run.run_baseline()

    m = io.read_tsv(str(tmp_path / "output" / "matching_results.tsv"))
    c = io.read_tsv(str(tmp_path / "output" / "candidate_pairs.tsv"))
    assert list(m["source1_entity_id"]) == list(c["source1_entity_id"]) == ["S1-1", "S1-2", "S1-3"]
    matches = dict(zip(m["source1_entity_id"], m["matched_entity_ids"]))
    assert matches == {"S1-1": "S2-1", "S1-2": "S3-1", "S1-3": ""}
    for mids, cids in zip(m["matched_entity_ids"], c["candidate_entity_ids"]):
        assert set(filter(None, mids.split(","))) <= set(cids.split(","))
