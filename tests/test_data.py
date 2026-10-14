import json

import pandas as pd
import pytest

from phishshield.data import prepare_dataset, split_dataset, split_summary, training_tld_statistics


def test_original_uci_labels_and_conflicts_are_explicit(tmp_path):
    path = tmp_path / "urls.csv"
    pd.DataFrame({
        "URL": ["https://EXAMPLE.com/a", "https://example.com/a", "https://conflict.example/",
                "https://CONFLICT.example/", "http://verify.example/login", "javascript:alert(1)", None],
        "label": [1, 1, 0, 1, 0, 0, 1],
        "unusable_webpage_feature": [999] * 7,
    }).to_csv(path, index=False)
    result = prepare_dataset(path)
    labels = dict(zip(result.frame["url"], result.frame["label"]))
    assert labels == {"https://example.com/a": 0, "http://verify.example/login": 1}
    assert result.summary["conflicting_label_rows_removed"] == 2
    assert result.summary["duplicate_rows_removed"] == 1
    assert result.summary["invalid_url_rows"] == 1
    assert result.summary["missing_url_or_label_rows"] == 1
    assert len(result.checksum) == 64
    assert "unusable_webpage_feature" not in result.frame.columns


def _domain_dataset():
    records = []
    for index in range(80):
        for page in range(3):
            records.append({"row_id": len(records), "url": f"https://sub{page}.site{index}.example/page",
                            "domain_group": f"site{index}.example", "label": index % 2, "tld": "example"})
    return pd.DataFrame(records)


def test_splits_keep_entire_domains_together_and_are_reproducible():
    frame = _domain_dataset()
    split = split_dataset(frame, seed=42)
    repeated = split_dataset(frame, seed=42)
    pd.testing.assert_frame_equal(split, repeated)
    assert split.groupby("domain_group")["split"].nunique().eq(1).all()
    assert set(split["split"]) == {"train", "validation", "calibration", "test"}
    assert split.groupby("split")["label"].nunique().eq(2).all()
    assert sum(value["rows"] for value in split_summary(split).values()) == len(frame)
    for split_name, target in {"train": .6, "validation": .15, "calibration": .1, "test": .15}.items():
        assert abs(split["split"].eq(split_name).mean() - target) < .025


def test_tld_statistics_only_use_training_rows():
    frame = pd.DataFrame({"split": ["train", "train", "test", "validation"],
                          "tld": ["example", "example", "example", "test"], "label": [0, 1, 1, 1]})
    assert training_tld_statistics(frame) == {"example": {"count": 2, "phishing_rate": .5}}


def test_unknown_original_labels_fail(tmp_path):
    path = tmp_path / "urls.csv"
    pd.DataFrame({"URL": ["https://a.example/", "https://b.example/"], "label": [1, "phishing"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="original UCI labels"):
        prepare_dataset(path)


def test_insufficient_class_domains_fail_clearly():
    frame = _domain_dataset()
    frame.loc[frame["label"] == 1, "domain_group"] = "single.example"
    with pytest.raises(ValueError, match="four distinct domain groups"):
        split_dataset(frame)


def test_hannousse_labels_source_and_format_profile_are_explicit(tmp_path):
    from phishshield.data import HANNOUSSE_SOURCE
    path = tmp_path / "urls.csv"
    pd.DataFrame({
        "URL": ["http://docs.example.org/article?a=1", "https://verify.example/login", "https://mail.example.com"],
        "status": ["legitimate", "phishing", " LEGITIMATE "],
        "page_rank": [999, 999, 999],
    }).to_csv(path, index=False)
    result = prepare_dataset(path)
    assert result.source == HANNOUSSE_SOURCE
    assert result.summary["dataset_kind"] == "hannousse"
    assert result.frame["label"].tolist() == [0, 1, 0]
    assert result.summary["original_label_mapping"] == {"legitimate": "legitimate", "phishing": "phishing"}
    assert result.summary["url_format_profile_by_class"]["legitimate"]["nonroot_path"] == .5
    assert "page_rank" not in result.frame.columns


@pytest.mark.parametrize("statuses", [[0, 1], ["legitimate", "safe"]])
def test_hannousse_rejects_ambiguous_status_labels(tmp_path, statuses):
    path = tmp_path / "urls.csv"
    pd.DataFrame({"URL": ["https://a.example/", "https://b.example/"], "status": statuses}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Hannousse status labels"):
        prepare_dataset(path)


def test_dataset_rejects_two_label_columns(tmp_path):
    path = tmp_path / "urls.csv"
    pd.DataFrame({"URL": ["https://a.example/"], "label": [0], "status": ["phishing"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not both"):
        prepare_dataset(path)


def test_dataset_provenance_requires_csv_checksum_and_source_kind(tmp_path):
    from phishshield.data import file_sha256
    path = tmp_path / "urls.csv"
    pd.DataFrame({"URL": ["https://a.example/", "https://b.example/"], "status": ["legitimate", "phishing"]}).to_csv(path, index=False)
    metadata = path.with_suffix(".source.json")
    metadata.write_text(json.dumps({"dataset_kind": "hannousse", "csv_sha256": file_sha256(path), "mirror_revision": "fixture"}))
    assert prepare_dataset(path).summary["provenance"]["mirror_revision"] == "fixture"
    metadata.write_text(json.dumps({"dataset_kind": "hannousse", "csv_sha256": "stale"}))
    with pytest.raises(ValueError, match="provenance does not match"):
        prepare_dataset(path)
