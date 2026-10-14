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
DEFAULT_CSV = ROOT / "data" / "raw" / DATASET_FILENAME
DEFAULT_RELEASE = ROOT / "artifacts" / "release"


def download_dataset(destination: Path) -> Path:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhishShield local training, analysis, and MLflow demonstration")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the labeled UCI PhiUSIIL CSV")
    download.add_argument("--output", type=Path, default=DEFAULT_CSV)
    train = commands.add_parser("train", help="Compare three models, log MLflow runs, and export the champion")
    train.add_argument("--data", type=Path, default=DEFAULT_CSV)
    train.add_argument("--output", type=Path, default=DEFAULT_RELEASE)
    train.add_argument("--tracking-uri", default=None)
    train.add_argument("--max-rows", type=int, default=None, help="Optional reproducible sample for development; recorded in metadata")
    train.add_argument("--seed", type=int, default=42)
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
            download_dataset(args.output)
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
