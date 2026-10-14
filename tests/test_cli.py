"""CLI feedback and dataset archive handling with all network I/O mocked."""

from io import BytesIO
import json
from unittest.mock import patch
import zipfile

import joblib
import pandas as pd
import pytest
from sklearn import __version__ as sklearn_version
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from phishshield.cli import DATASET_FILENAME, DATASET_URL, DEFAULT_CSV, HANNOUSSE_FILENAME, download_dataset, main
from phishshield.features import FEATURE_VERSION, URLFeatureExtractor
from phishshield.inference import file_sha256


class DownloadResponse:
    def __init__(self, data=b"", failure=None):
        self.data = data
        self.failure = failure

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data[:10]
        if self.failure:
            raise self.failure
        yield self.data[10:]


def zipped_files(entries):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, contents in entries:
            archive.writestr(name, contents)
    return stream.getvalue()


@pytest.fixture
def local_release(tmp_path):
    directory = tmp_path / "release"
    directory.mkdir()
    pipeline = Pipeline([
        ("features", URLFeatureExtractor()),
        ("classifier", DummyClassifier(strategy="prior")),
    ]).fit(pd.DataFrame({"url": ["https://example.com/", "http://school.example/login"]}), [0, 1])
    model = directory / "model.joblib"
    joblib.dump(pipeline, model)
    manifest = {
        "feature_version": FEATURE_VERSION, "model_sha256": file_sha256(model),
        "dependency_versions": {"scikit-learn": sklearn_version},
        "threshold": 0.5, "model_version": "1", "run_id": "test-run", "algorithm": "DummyClassifier",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "known_urls.json").write_text(json.dumps([{
        "url": "https://example.com/", "phishing_probability": 0.5, "predicted_label": 1,
    }]), encoding="utf-8")
    return directory


def test_help_lists_project_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert all(command in output for command in ("download", "train", "analyze", "verify", "serve", "mlflow"))


def test_analyze_outputs_serializable_result_without_network(local_release, capsys):
    with patch("requests.get", side_effect=AssertionError("CLI inference must be offline")):
        assert main(["analyze", "https://example.com/", "--release", str(local_release)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["category"] == "Potential phishing"
    assert result["phishing_probability"] == 0.5
    assert result["model_version"] == "1"


def test_analyze_invalid_url_returns_clean_error(local_release, capsys):
    assert main(["analyze", "example.com", "--release", str(local_release)]) == 1
    output = capsys.readouterr()
    assert "Include http:// or https://" in output.err
    assert "Traceback" not in output.err
    assert output.out == ""


def test_verify_missing_release_returns_clean_error(tmp_path, capsys):
    assert main(["verify", "--release", str(tmp_path / "missing")]) == 1
    output = capsys.readouterr()
    assert "train first" in output.err
    assert "Traceback" not in output.err


def test_verify_existing_release_outputs_json(local_release, capsys):
    assert main(["verify", "--release", str(local_release)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert result["examples_checked"] == 1


def test_download_reads_expected_basename_without_extracting_other_files(tmp_path):
    contents = b"URL,label\nhttps://example.com/,1\n"
    archive = zipped_files([
        ("nested/" + DATASET_FILENAME, contents),
        ("../../unexpected.txt", b"not extracted"),
    ])
    destination = tmp_path / "dataset" / DATASET_FILENAME
    with patch("requests.get", return_value=DownloadResponse(archive)) as request:
        assert download_dataset(destination, dataset="phiusiil") == destination
    request.assert_called_once_with(DATASET_URL, stream=True, timeout=(15, 120))
    assert destination.read_bytes() == contents
    assert sorted(path.name for path in destination.parent.iterdir()) == [DATASET_FILENAME]
    assert not (tmp_path / "unexpected.txt").exists()


def test_existing_dataset_is_not_downloaded_again(tmp_path):
    destination = tmp_path / DATASET_FILENAME
    destination.write_text("existing data", encoding="utf-8")
    with patch("requests.get", side_effect=AssertionError("Existing dataset should be reused")):
        assert download_dataset(destination, dataset="phiusiil") == destination
    assert destination.read_text(encoding="utf-8") == "existing data"


@pytest.mark.parametrize("entries", [
    [("wrong.csv", "URL,label\n")],
    [("one/" + DATASET_FILENAME, "first"), ("two/" + DATASET_FILENAME, "second")],
])
def test_download_rejects_missing_or_ambiguous_csv_and_cleans_partials(tmp_path, entries):
    destination = tmp_path / DATASET_FILENAME
    with patch("requests.get", return_value=DownloadResponse(zipped_files(entries))):
        with pytest.raises(ValueError, match="expected CSV"):
            download_dataset(destination, dataset="phiusiil")
    assert list(tmp_path.iterdir()) == []


def test_download_network_failure_cleans_partials(tmp_path):
    destination = tmp_path / DATASET_FILENAME
    response = DownloadResponse(b"partially downloaded", failure=OSError("connection interrupted"))
    with patch("requests.get", return_value=response):
        with pytest.raises(OSError, match="connection interrupted"):
            download_dataset(destination, dataset="phiusiil")
    assert list(tmp_path.iterdir()) == []


def test_download_failure_cli_has_no_traceback_and_cleans_partials(tmp_path, capsys):
    destination = tmp_path / DATASET_FILENAME
    with patch("requests.get", return_value=DownloadResponse(b"invalid ZIP bytes")):
        assert main(["download", "--dataset", "phiusiil", "--output", str(destination)]) == 1
    output = capsys.readouterr()
    assert "PhishShield:" in output.err
    assert "Traceback" not in output.err
    assert list(tmp_path.iterdir()) == []


def parquet_bytes(urls, statuses):
    stream = BytesIO()
    pd.DataFrame({"url": urls, "status": statuses, "page_rank": [999] * len(urls)}).to_parquet(stream, index=False)
    return stream.getvalue()


def test_default_download_recombines_pinned_parquet_urls_and_records_provenance(tmp_path):
    from phishshield.data import HANNOUSSE_MIRROR_BASE, HANNOUSSE_MIRROR_REVISION, prepare_dataset
    destination = tmp_path / HANNOUSSE_FILENAME
    responses = [
        DownloadResponse(parquet_bytes(["http://docs.example/a"], ["legitimate"])),
        DownloadResponse(parquet_bytes(["https://login.example/b"], ["phishing"])),
    ]
    with patch("requests.get", side_effect=responses) as request:
        assert download_dataset(destination) == destination
    assert [call.args[0] for call in request.call_args_list] == [HANNOUSSE_MIRROR_BASE + "train.parquet", HANNOUSSE_MIRROR_BASE + "test.parquet"]
    data = pd.read_csv(destination)
    assert list(data.columns) == ["URL", "status"]
    assert data["status"].tolist() == ["legitimate", "phishing"]
    result = prepare_dataset(destination)
    assert result.summary["provenance"]["mirror_revision"] == HANNOUSSE_MIRROR_REVISION
    assert len(result.summary["provenance"]["source_files"]) == 2
    assert sorted(file.name for file in tmp_path.iterdir()) == [destination.name, destination.with_suffix(".source.json").name]


def test_hannousse_failed_second_download_cleans_every_partial(tmp_path):
    destination = tmp_path / HANNOUSSE_FILENAME
    responses = [DownloadResponse(parquet_bytes(["https://a.example"], ["legitimate"])),
                 DownloadResponse(b"partial", failure=OSError("connection interrupted"))]
    with patch("requests.get", side_effect=responses):
        with pytest.raises(OSError, match="connection interrupted"):
            download_dataset(destination)
    assert list(tmp_path.iterdir()) == []


def test_hannousse_download_rejects_unknown_labels_and_cleans_every_partial(tmp_path):
    destination = tmp_path / HANNOUSSE_FILENAME
    with patch("requests.get", return_value=DownloadResponse(parquet_bytes(["https://a.example"], ["safe"]))):
        with pytest.raises(ValueError, match="unexpected URL/status labels"):
            download_dataset(destination)
    assert list(tmp_path.iterdir()) == []


def test_download_cli_defaults_to_hannousse_and_phiusiil_selects_own_filename():
    assert DEFAULT_CSV.name == HANNOUSSE_FILENAME
    with patch("phishshield.cli.download_dataset") as downloader:
        assert main(["download"]) == 0
        downloader.assert_called_once_with(DEFAULT_CSV, dataset="hannousse")
    with patch("phishshield.cli.download_dataset") as downloader:
        assert main(["download", "--dataset", "phiusiil"]) == 0
        assert downloader.call_args.args[0].name == DATASET_FILENAME
        assert downloader.call_args.kwargs["dataset"] == "phiusiil"
