"""Documented label mapping, conservative cleaning, and domain-disjoint splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from phishshield.features import URLValidationError, parse_url


DATASET_SOURCE = (
    "https://archive.ics.uci.edu/dataset/967/"
    "phiusiil%2Bphishing%2Burl%2Bwebsite%2Bdataset"
)
HANNOUSSE_SOURCE = "https://data.mendeley.com/datasets/c2gw7fy2j4/3"
HANNOUSSE_MIRROR_REVISION = "1375d450d4540205ac11781b807249c9509739cd"
HANNOUSSE_MIRROR_BASE = (
    "https://huggingface.co/datasets/pirocheto/phishing-url/resolve/"
    + HANNOUSSE_MIRROR_REVISION + "/data/"
)
HANNOUSSE_MIRROR_FILES = ("train.parquet", "test.parquet")
SPLIT_FRACTIONS = {"train": 0.60, "validation": 0.15, "calibration": 0.10, "test": 0.15}


@dataclass
class PreparedDataset:
    frame: pd.DataFrame
    summary: dict
    checksum: str
    source: str = DATASET_SOURCE


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(
    csv_path: str | Path, *, max_rows: int | None = None, seed: int = 42
) -> PreparedDataset:
    """Read raw URL text and source-specific labels without webpage features.

    Hannousse's ``status`` uses phishing/legitimate strings. Original UCI
    ``label`` uses 0=phishing and 1=legitimate. Both become internal 1=phishing,
    0=legitimate. All conflicting normalized URLs are removed before sampling.
    """
    path = Path(csv_path).expanduser().resolve()
    if max_rows is not None and max_rows < 40:
        raise ValueError("max_rows must be at least 40 for four evaluation partitions.")
    raw = pd.read_csv(
        path,
        usecols=lambda column: str(column).strip().casefold() in {"url", "label", "status"},
        low_memory=False,
    )
    raw.columns = [str(column).strip().casefold() for column in raw.columns]
    columns = set(raw.columns)
    if columns == {"url", "status"}:
        dataset_kind, source, label_column = "hannousse", HANNOUSSE_SOURCE, "status"
        original_mapping = {"legitimate": "legitimate", "phishing": "phishing"}
    elif columns == {"url", "label"}:
        dataset_kind, source, label_column = "phiusiil", DATASET_SOURCE, "label"
        original_mapping = {"0": "phishing", "1": "legitimate"}
    else:
        raise ValueError("The CSV must contain URL and either Hannousse status or original UCI label columns, not both.")
    raw_count = len(raw)
    missing_mask = raw["url"].isna() | raw[label_column].isna()
    raw = raw.loc[~missing_mask].copy()
    if dataset_kind == "hannousse":
        statuses = raw["status"].astype(str).str.strip().str.casefold()
        if not statuses.isin(["legitimate", "phishing"]).all():
            raise ValueError("Expected Hannousse status labels: legitimate and phishing.")
        internal_labels = statuses.map({"legitimate": 0, "phishing": 1})
    else:
        original_labels = pd.to_numeric(raw["label"], errors="coerce")
        if not original_labels.isin([0, 1]).all():
            raise ValueError("Expected original UCI labels: 1=legitimate and 0=phishing.")
        internal_labels = 1 - original_labels.astype(int)
    checksum = file_sha256(path)
    provenance = {"dataset_kind": dataset_kind, "source": source, "label_column": label_column,
                  "csv_sha256": checksum}
    metadata_path = path.with_suffix(".source.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("dataset_kind") != dataset_kind or metadata.get("csv_sha256") != checksum:
            raise ValueError("Dataset provenance does not match this CSV; download it again or supply its correct metadata.")
        provenance.update(metadata)

    records = []
    invalid_urls = 0
    for row_id, row in raw.iterrows():
        try:
            parsed = parse_url(str(row["url"]))
        except URLValidationError:
            invalid_urls += 1
            continue
        records.append({
            "row_id": int(row_id),
            "url": parsed.normalized_url,
            "label": int(internal_labels.loc[row_id]),
            "domain_group": parsed.registered_domain or parsed.hostname,
            "tld": parsed.tld,
            "_https": int(parsed.scheme == "https"),
            "_nonroot_path": int(bool(parsed.path)),
            "_query": int(bool(parsed.query)),
            "_www": int(parsed.hostname.startswith("www.")),
        })
    if not records:
        raise ValueError("No usable HTTP(S) URLs remain after validation.")
    frame = pd.DataFrame.from_records(records)
    conflicting = frame.groupby("url")["label"].nunique()
    conflicting_urls = set(conflicting.index[conflicting > 1])
    conflict_mask = frame["url"].isin(conflicting_urls)
    conflicting_rows = int(conflict_mask.sum())
    frame = frame.loc[~conflict_mask].copy()
    duplicate_rows = int(frame.duplicated("url").sum())
    frame = frame.drop_duplicates("url", keep="first")
    usable_rows = len(frame)
    if frame["label"].nunique() != 2:
        raise ValueError("The cleaned dataset must contain both legitimate and phishing URLs.")
    if max_rows is not None and len(frame) > max_rows:
        frame, _ = train_test_split(
            frame, train_size=max_rows, stratify=frame["label"], random_state=seed
        )
    frame = frame.sort_values("row_id").reset_index(drop=True)
    format_profile = {
        ("phishing" if label else "legitimate"): {
            name.removeprefix("_"): float(group[name].mean())
            for name in ("_https", "_nonroot_path", "_query", "_www")
        }
        for label, group in frame.groupby("label", sort=True)
    }
    frame = frame.drop(columns=["_https", "_nonroot_path", "_query", "_www"])
    summary = {
        "dataset_source": source,
        "dataset_kind": dataset_kind,
        "provenance": provenance,
        "url_format_profile_by_class": format_profile,
        "original_label_mapping": original_mapping,
        "internal_label_mapping": {"0": "legitimate", "1": "phishing"},
        "raw_rows": raw_count,
        "missing_url_or_label_rows": int(missing_mask.sum()),
        "invalid_url_rows": invalid_urls,
        "conflicting_normalized_urls": len(conflicting_urls),
        "conflicting_label_rows_removed": conflicting_rows,
        "duplicate_rows_removed": duplicate_rows,
        "usable_rows_before_sampling": usable_rows,
        "selected_rows": len(frame),
        "domain_groups": int(frame["domain_group"].nunique()),
        "class_counts": {"legitimate": int((frame["label"] == 0).sum()), "phishing": int((frame["label"] == 1).sum())},
        "sampling_seed": seed,
    }
    return PreparedDataset(frame=frame, summary=summary, checksum=checksum, source=source)


def split_dataset(frame: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Split whole registrable domains approximately 60/15/10/15.

    GroupShuffleSplit targets fractions of groups, so actual row fractions
    are reported. Deterministic retries only ensure each partition contains
    both labels; no model predictions or performance enter split selection.
    """
    if frame["domain_group"].nunique() < 8:
        raise ValueError("At least eight distinct domain groups are needed for the four splits.")
    for label in (0, 1):
        if frame.loc[frame["label"] == label, "domain_group"].nunique() < 4:
            raise ValueError("Each class needs URLs from at least four distinct domain groups.")
    for attempt in range(50):
        split_seed = seed + attempt
        train_idx, remaining_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=0.40, random_state=split_seed
        ).split(frame, groups=frame["domain_group"]))
        remaining = frame.iloc[remaining_idx]
        val_idx, later_idx = next(GroupShuffleSplit(
            n_splits=1, train_size=0.375, random_state=split_seed + 1000
        ).split(remaining, groups=remaining["domain_group"]))
        later = remaining.iloc[later_idx]
        cal_idx, test_idx = next(GroupShuffleSplit(
            n_splits=1, train_size=0.40, random_state=split_seed + 2000
        ).split(later, groups=later["domain_group"]))
        indices = {
            "train": frame.index[train_idx],
            "validation": remaining.index[val_idx],
            "calibration": later.index[cal_idx],
            "test": later.index[test_idx],
        }
        if any(frame.loc[index, "label"].nunique() != 2 for index in indices.values()):
            continue
        result = frame.copy()
        result["split"] = ""
        for name, index in indices.items():
            result.loc[index, "split"] = name
        if result["split"].eq("").any() or result.groupby("domain_group")["split"].nunique().max() != 1:
            raise RuntimeError("Domain-disjoint split verification failed.")
        result.attrs["split_seed"] = split_seed
        return result
    raise ValueError("Could not make four domain-disjoint splits containing both classes. Use more domains.")


def split_summary(frame: pd.DataFrame) -> dict:
    return {
        name: {
            "rows": int(len(partition)),
            "row_fraction": float(len(partition) / len(frame)),
            "groups": int(partition["domain_group"].nunique()),
            "legitimate_rows": int((partition["label"] == 0).sum()),
            "phishing_rows": int((partition["label"] == 1).sum()),
        }
        for name, partition in frame.groupby("split", sort=True)
    }


def training_tld_statistics(frame: pd.DataFrame) -> dict[str, dict[str, int | float]]:
    training = frame.loc[frame["split"] == "train"]
    return {
        str(suffix): {"count": int(len(group)), "phishing_rate": float(group["label"].mean())}
        for suffix, group in training.groupby("tld", sort=True)
        if suffix
    }
