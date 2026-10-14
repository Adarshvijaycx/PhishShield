"""Commands for dataset download, reproducible training, and local review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

ROOT = Path(__file__).resolve().parents[2]
DATASET_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip"
DATASET_FILENAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
HANNOUSSE_FILENAME = "Hannousse_Phishing_URL_Dataset.csv"
DEFAULT_CSV = ROOT / "data" / "raw" / HANNOUSSE_FILENAME
DEFAULT_RELEASE = ROOT / "artifacts" / "release"


def _download_phiusiil(destination: Path) -> Path:
    import requests

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"Dataset already exists: {destination}")
        return destination
    archive = destination.parent / "phiusiil.zip.part"
    temporary_csv = destination.with_suffix(".csv.part")
    try:
        print("Downloading the documented UCI dataset; submitted websites are never visited.", flush=True)
        with requests.get(DATASET_URL, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with archive.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
        with zipfile.ZipFile(archive) as zipped:
            candidates = [name for name in zipped.namelist() if Path(name).name == DATASET_FILENAME]
            if len(candidates) != 1:
                raise ValueError("The UCI archive does not contain the expected CSV.")
            with zipped.open(candidates[0]) as source, temporary_csv.open("wb") as target:
                shutil.copyfileobj(source, target)
        temporary_csv.replace(destination)
    finally:
        archive.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)
    print(f"Dataset saved: {destination}")
    return destination


def _download_hannousse(destination: Path) -> Path:
    """Recombine the pinned mirror splits, keeping only raw URLs and labels.

    The mirror's old random partitions are discarded. Training makes fresh
    domain-disjoint partitions after normalization and deduplication.
    """
    import pandas as pd
    import requests
    from phishshield.data import (
        HANNOUSSE_SOURCE, HANNOUSSE_MIRROR_REVISION, HANNOUSSE_MIRROR_BASE,
        HANNOUSSE_MIRROR_FILES, file_sha256,
    )

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"Dataset already exists: {destination}")
        return destination
    temporary_csv = destination.with_suffix(".csv.part")
    metadata = destination.with_suffix(".source.json")
    temporary_metadata = metadata.with_suffix(".json.part")
    downloads = []
    try:
        print("Downloading the licensed Hannousse URL dataset from a pinned mirror; submitted websites are never visited.", flush=True)
        frames, files = [], []
        for filename in HANNOUSSE_MIRROR_FILES:
            temporary = destination.parent / ("hannousse-" + filename + ".part")
            downloads.append(temporary)
            url = HANNOUSSE_MIRROR_BASE + filename
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(chunk)
            projected = pd.read_parquet(temporary, columns=["url", "status"])
            statuses = projected["status"].astype(str).str.strip().str.casefold()
            if projected["url"].isna().any() or not statuses.isin(["phishing", "legitimate"]).all():
                raise ValueError("The Hannousse mirror contains unexpected URL/status labels.")
            frames.append(projected)
            files.append({"url": url, "sha256": file_sha256(temporary), "rows": len(projected)})
        data = pd.concat(frames, ignore_index=True).rename(columns={"url": "URL"})
        data.to_csv(temporary_csv, index=False)
        provenance = {
            "dataset_kind": "hannousse", "source": HANNOUSSE_SOURCE,
            "license": "CC BY 4.0", "authors": ["Abdelhakim Hannousse", "Salima Yahiouche"],
            "mirror": "https://huggingface.co/datasets/pirocheto/phishing-url",
            "mirror_revision": HANNOUSSE_MIRROR_REVISION, "source_files": files,
            "changes": "Recombined mirror train/test files; retained only raw URL and status; original partitions discarded.",
            "csv_sha256": file_sha256(temporary_csv), "raw_rows": len(data),
        }
        temporary_metadata.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        temporary_csv.replace(destination)
        temporary_metadata.replace(metadata)
    finally:
        for temporary in downloads:
            temporary.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    print(f"Dataset saved: {destination}")
    return destination


def download_dataset(destination: Path, dataset: str = "hannousse") -> Path:
    """Download the chosen source without contacting any URLs in the dataset."""
    if dataset == "hannousse":
        return _download_hannousse(destination)
    if dataset == "phiusiil":
        return _download_phiusiil(destination)
    raise ValueError("Choose a documented dataset: hannousse or phiusiil.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhishShield local training, analysis, and MLflow demonstration")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download a documented labeled raw-URL dataset")
    download.add_argument("--dataset", choices=["hannousse", "phiusiil"], default="hannousse")
    download.add_argument("--output", type=Path, default=None)
    train = commands.add_parser("train", help="Compare three models, log MLflow runs, and export the champion")
    train.add_argument("--data", type=Path, default=DEFAULT_CSV)
    train.add_argument("--output", type=Path, default=DEFAULT_RELEASE)
    train.add_argument("--tracking-uri", default=None)
    train.add_argument("--max-rows", type=int, default=None, help="Optional reproducible sample for development; recorded in metadata")
    train.add_argument("--seed", type=int, default=43)
    train.add_argument("--importance-samples", type=int, default=2000)
    for command in ["analyze", "verify"]:
        sub = commands.add_parser(command, help="Analyze a URL" if command == "analyze" else "Check release integrity and regression fixtures")
        sub.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
        if command == "analyze":
            sub.add_argument("url")
    serve = commands.add_parser("serve", help="Start the local Streamlit analyzer")
    serve.add_argument("--port", type=int, default=8501)
    serve.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    tracking = commands.add_parser("mlflow", help="Start the local MLflow UI and registry")
    tracking.add_argument("--port", type=int, default=5001)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            output = args.output or (ROOT / "data" / "raw" / (HANNOUSSE_FILENAME if args.dataset == "hannousse" else DATASET_FILENAME))
            download_dataset(output, dataset=args.dataset)
        elif args.command == "train":
            from phishshield.training import train_project
            result = train_project(args.data, args.output, tracking_uri=args.tracking_uri, max_rows=args.max_rows, seed=args.seed, importance_samples=args.importance_samples)
            print(json.dumps({key: result.get(key) for key in ["algorithm", "model_version", "run_id", "metrics"]}, indent=2))
        elif args.command in {"analyze", "verify"}:
            from phishshield.inference import analyze_url, load_release, verify_release
            result = analyze_url(args.url, load_release(args.release)).to_dict() if args.command == "analyze" else verify_release(args.release)
            print(json.dumps(result, indent=2))
        elif args.command == "serve":
            environment = os.environ.copy()
            environment["PHISHSHIELD_RELEASE_DIR"] = str(args.release.resolve())
            return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.address=127.0.0.1", f"--server.port={args.port}", "--server.headless=true", "--browser.gatherUsageStats=false"], env=environment, cwd=ROOT)
        elif args.command == "mlflow":
            return subprocess.call([sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", str(args.port), "--backend-store-uri", f"sqlite:///{ROOT / 'mlflow.db'}", "--artifacts-destination", str(ROOT / "mlartifacts")], cwd=ROOT)
    except KeyboardInterrupt:
        print("Stopped PhishShield.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as exc:
        print(f"PhishShield: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
