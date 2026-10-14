"""PhishShield's local, release-backed Streamlit demonstration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from phishshield.features import URLValidationError
from phishshield.inference import ModelLoadError, analyze_url, load_release


APP_ROOT = Path(__file__).resolve().parent
EXAMPLES = {
    "Simple HTTPS": "https://www.example.com/",
    "Account keywords": "http://secure-account.example.org/login/verify?account=update",
    "Misleading address": "http://example.com@192.0.2.10/verify-account",
}
RISK_COLORS = {"Low": "#28D5B6", "Medium": "#F3BD58", "High": "#FA7B8B"}


def _inject_style() -> None:
    # Only static markup is rendered as HTML. Submitted URLs stay in native widgets.
    st.markdown(
        """
        <style>
        .stApp { background: #08111f; }
        .block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background: #0d1828; border-right: 1px solid #223148; }
        [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
        h1,h2,h3 { letter-spacing: -.035em; }
        [data-testid="stMetricValue"] { font-weight: 650; }
        [data-testid="stMetricLabel"] { color: #9caec7; }
        [data-testid="stForm"] { background: #101d30; border-color: #2b405a; padding: 1.2rem; }
        [data-testid="stTextInput"] input { min-height: 3rem; }
        [data-testid="stTabs"] button p { font-weight: 650; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.8rem; border-bottom: 1px solid #223148; }
        [data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #243650; }
        .ps-hero { padding: 1rem 0 1.3rem; }
        .ps-eyebrow { font-size: .76rem; text-transform: uppercase; letter-spacing: .18em; color: #28d5b6; font-weight: 750; }
        .ps-hero h1 { font-size: clamp(2rem,4vw,3.15rem); line-height: 1.12; margin: .8rem 0 .8rem; color: #f2f6ff; }
        .ps-hero h1 span { color: #28d5b6; }
        .ps-hero p { color: #a9b9d0; max-width: 710px; font-size: 1.02rem; line-height: 1.7; margin-bottom: .4rem; }
        .ps-brand { display:flex; align-items:center; gap:.65rem; font-weight:800; letter-spacing:-.04em; font-size:1.48rem; color:#f2f6ff; }
        .ps-brand svg { flex:none; }
        .ps-result-label { font-size:.74rem; text-transform:uppercase; letter-spacing:.14em; color:#9caec7; margin-top:.25rem; }
        .ps-badge { display:inline-block; padding:.35rem .7rem; border:1px solid currentColor; border-radius:999px; font-size:.78rem; font-weight:650; margin-bottom:.65rem; }
        .ps-footnote { border-top:1px solid #223148; margin-top:2rem; padding-top:1rem; color:#9caec7; font-size:.83rem; line-height:1.6; }
        @media(max-width:640px) { .block-container {padding-top:1.4rem;} .ps-hero {padding-top:0;} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _release_fingerprint(release_dir: Path) -> tuple:
    """Invalidate the resource cache when the exported release changes."""
    fingerprint = []
    for filename in ("manifest.json", "model.joblib"):
        try:
            stat = (release_dir / filename).stat()
            fingerprint.append((filename, stat.st_mtime_ns, stat.st_size))
        except OSError:
            fingerprint.append((filename, None, None))
    return tuple(fingerprint)


@st.cache_resource(show_spinner=False)
def _load_cached_release(release_directory: str, fingerprint: tuple):
    return load_release(Path(release_directory))


def _choose_example(url: str) -> None:
    st.session_state["url_input"] = url
    st.session_state.pop("prediction", None)
    st.session_state.pop("analysis_error", None)


def _gauge(score: float, band: str) -> go.Figure:
    color = RISK_COLORS.get(band, "#28D5B6")
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".1f", "font": {"size": 43, "color": "#F2F6FF"}},
            title={"text": "Model phishing estimate", "font": {"size": 15, "color": "#A9B9D0"}},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 30, 70, 100], "tickcolor": "#6B829F"},
                "bar": {"color": color, "thickness": .26},
                "bgcolor": "#101D30",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#153E40"},
                    {"range": [30, 70], "color": "#4A3F2B"},
                    {"range": [70, 100], "color": "#482B3D"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "thickness": .85, "value": score},
            },
        )
    )
    figure.update_layout(
        height=250, margin={"l": 25, "r": 25, "t": 55, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#A9B9D0", "family": "sans-serif"},
    )
    return figure


def _sidebar(bundle, model_error: str | None) -> None:
    with st.sidebar:
        st.markdown(
            """<div class="ps-brand"><svg width="31" height="36" viewBox="0 0 31 36" fill="none" aria-hidden="true"><path d="M15.5 2 29 7v11c0 7.5-5.4 12.7-13.5 16C7.4 30.7 2 25.5 2 18V7L15.5 2Z" fill="#153e40" stroke="#28d5b6" stroke-width="2"/><path d="m9 17 4.2 4.2L22 12" stroke="#28d5b6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>PhishShield</div>""",
            unsafe_allow_html=True,
        )
        st.caption("URL & phishing risk intelligence")
        st.divider()
        if bundle is None:
            st.warning("Model release unavailable")
            st.caption("Train and export a release to enable real predictions.")
        else:
            manifest = bundle.manifest
            st.success("Model ready")
            st.caption("ACTIVE RELEASE")
            st.write(f"**{manifest.get('algorithm', 'Trained classifier')}**")
            st.caption(f"Registry version {manifest.get('model_version', 'unknown')} · features {manifest.get('feature_version', 'unknown')}")
            test_rows = manifest.get("test_rows")
            if test_rows is not None:
                st.caption(f"Evaluated on {int(test_rows):,} held-out URLs")
            with st.expander("Release identity"):
                st.text(f"Model: {manifest.get('model_name', 'PhishShield')}")
                st.text(f"Version: {manifest.get('model_version', 'unknown')}")
                st.text(f"Run: {manifest.get('run_id', 'unknown')}")
                st.caption(f"Created: {manifest.get('created_at', 'unknown')}")
        st.divider()
        st.caption("LOCAL EXPERIMENT TRACKING")
        st.link_button("Open MLflow ↗", os.environ.get("PHISHSHIELD_MLFLOW_URL", "http://127.0.0.1:5001"), width="stretch")
        st.caption("Start the local MLflow server first. The analyzer loads the exported model directly.")
        with st.expander("How it works"):
            st.write("1. Parse the URL text.\n2. Extract its structural features.\n3. Run the calibrated classifier.\n4. Show the estimated risk and observed indicators.")
        st.divider()
        st.caption("ACADEMIC PROJECT · LOCAL DEMO")
        st.caption("No website requests. No submitted URL history.")


def _missing_release(error: str) -> None:
    st.warning("A trained model release is needed before analysis.")
    st.write("Download the labeled dataset, train the candidate models, and export the MLflow champion:")
    st.code(".venv/bin/python -m phishshield.cli download\n.venv/bin/python -m phishshield.cli train", language="bash")
    with st.expander("Release loading details"):
        st.text(error)
    st.caption("Predictions will appear here after a valid release is available. No simulated scores are shown.")


def _show_result(result) -> None:
    with st.container(border=True):
        details, gauge = st.columns([1.25, 1], gap="large")
        with details:
            st.markdown('<div class="ps-result-label">Analysis result</div>', unsafe_allow_html=True)
            st.subheader(result.category)
            band = result.assessment_band
            # The band is fixed by the inference API, never derived from user markup.
            color = RISK_COLORS.get(band, "#28D5B6")
            safe_band = band if band in RISK_COLORS else "Unknown"
            badge = f"{safe_band} caution · domain warning" if result.assessment_basis == "domain_warning" else f"{safe_band} model risk"
            st.markdown(f'<span class="ps-badge" style="color:{color}">{badge}</span>', unsafe_allow_html=True)
            if result.domain_signals:
                st.error("The domain resembles a known brand but is outside its comparison domains. Verify the address independently before opening it.")
            st.caption("EFFECTIVE HOSTNAME")
            st.text(result.hostname)
            st.caption("LAST ANALYZED URL")
            st.code(result.normalized_url, language=None, wrap_lines=True)
        with gauge:
            if result.domain_signals:
                st.write("**Domain warning takes priority**")
                st.caption("A low model estimate does not clear this domain warning. High caution is a review rule, not a measured phishing probability.")
                with st.expander("View the unmodified model estimate"):
                    st.plotly_chart(_gauge(result.risk_score, result.risk_band), width="stretch", config={"displayModeBar": False}, key="risk_gauge")
                    st.caption(f"Model only: {result.model_category} · {result.risk_band} · {result.risk_score:.1f}/100")
            else:
                st.plotly_chart(_gauge(result.risk_score, result.risk_band), width="stretch", config={"displayModeBar": False}, key="risk_gauge")
                st.caption("Low <30 · Medium 30–<70 · High ≥70")
        st.caption("Model score = calibrated P(phishing) × 100 on historical URL data. The model class uses a 50% threshold. Independent domain warnings require review and leave this probability unchanged.")

    if result.domain_signals:
        st.subheader("Domain identity warnings")
        for signal in result.domain_signals:
            with st.container(border=True):
                st.write(f"**{signal['title']}**")
                st.caption(signal["description"])
        st.caption("The comparison catalogue and supported lookalike patterns are limited. A warning is not proof of phishing; its absence does not verify ownership or safety.")

    st.subheader("Suspicious indicators")
    st.caption("Observed URL patterns. These indicators are context, not an explanation of this individual prediction.")
    if not result.indicators:
        st.info("No additional structural warning patterns were found.")
        st.caption("A URL with no listed indicators can still be phishing.")
    else:
        columns = st.columns(2, gap="medium")
        for index, indicator in enumerate(result.indicators):
            with columns[index % 2], st.container(border=True):
                st.write(f"**{indicator['title']}**")
                st.caption(indicator["description"])
    with st.expander("URL breakdown & extracted features"):
        st.caption("REGISTERED DOMAIN")
        st.text(result.registered_domain or "Not applicable")
        features = pd.DataFrame({"Feature": list(result.features), "Value": list(result.features.values())})
        st.dataframe(features, width="stretch", hide_index=True)
    st.caption(f"Analyzed with {result.algorithm} · model version {result.model_version}")


def _analyzer(bundle, model_error: str | None) -> None:
    st.markdown("### Check a URL")
    st.caption("Paste the complete address, including http:// or https://. Analysis uses URL text only.")
    example_columns = st.columns(len(EXAMPLES))
    for column, (label, url) in zip(example_columns, EXAMPLES.items()):
        column.button(label, key=f"example_{label}", on_click=_choose_example, args=(url,), width="stretch")
    st.caption("Example buttons use reserved domains and documentation IPs. They demonstrate patterns, not verified phishing labels.")
    with st.form("url_analysis_form", clear_on_submit=False):
        url = st.text_input("URL to analyze", key="url_input", placeholder="https://www.example.com/", help="Use a complete address, such as https://www.example.com/. The tool never opens or fetches it.")
        submitted = st.form_submit_button("Analyze URL", type="primary", width="stretch", disabled=bundle is None)
    if bundle is None:
        _missing_release(model_error or "No valid model release was found.")
        return
    if submitted:
        st.session_state.pop("prediction", None)
        st.session_state.pop("analysis_error", None)
        try:
            with st.spinner("Checking URL patterns…"):
                st.session_state["prediction"] = analyze_url(url, bundle)
        except URLValidationError as exc:
            st.session_state["analysis_error"] = str(exc)
        except (ModelLoadError, ValueError, TypeError):
            st.session_state["analysis_error"] = "The loaded model could not analyze this URL. Verify the release bundle and try again."
    if st.session_state.get("analysis_error"):
        st.error(st.session_state["analysis_error"])
    if st.session_state.get("prediction") is not None:
        _show_result(st.session_state["prediction"])
    else:
        with st.container(border=True):
            st.write("**Understand the address before acting**")
            st.caption("Your result will include the model's category, estimated risk, effective hostname, and observable URL indicators.")


def _read_artifact_table(bundle, filename: str) -> pd.DataFrame | None:
    path = bundle.release_dir / filename
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        st.warning(f"The {filename} artifact could not be read. Re-export the model release.")
        return None


def _importance_chart(frame: pd.DataFrame) -> go.Figure:
    top = frame.nlargest(10, "importance_mean").sort_values("importance_mean")
    figure = go.Figure(go.Bar(
        x=top["importance_mean"], y=top["feature"].str.replace("_", " "), orientation="h",
        marker={"color": "#28D5B6", "cornerradius": 4},
        error_x={"type": "data", "array": top["importance_std"], "color": "#91ACBF", "thickness": 1},
        hovertemplate="%{y}<br>Mean AP decrease: %{x:.4f}<extra></extra>",
    ))
    figure.update_layout(
        height=370, margin={"l": 5, "r": 20, "t": 10, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#A9B9D0"}, xaxis={"title": "Decrease in average precision", "gridcolor": "#223148", "zerolinecolor": "#40546F"},
        yaxis={"title": None, "automargin": True},
    )
    return figure


def _artifact_image(bundle, filename: str, caption: str) -> None:
    path = bundle.release_dir / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"{caption} is not included in this release.")


def _insights(bundle, model_error: str | None) -> None:
    st.markdown("### Evidence behind the model")
    st.caption("Real experiment results from the exported MLflow release. Candidate selection and final test evaluation use separate partitions.")
    st.caption("Historical dataset results describe this evaluation only. The current corpus includes legitimate paths and query strings; live reputation and page content are not checked.")
    if bundle is None:
        _missing_release(model_error or "No valid model release was found.")
        return
    manifest = bundle.manifest
    metrics = manifest.get("metrics", {})
    st.markdown("#### Calibrated winner · held-out test set")
    metric_columns = st.columns(4)
    for column, (key, label) in zip(metric_columns, (("recall", "Phishing recall"), ("precision", "Phishing precision"), ("f1", "Phishing F1"), ("roc_auc", "ROC AUC"))):
        value = metrics.get(key)
        column.metric(label, f"{float(value) * 100:.2f}%" if value is not None else "Unavailable")
    st.caption(f"{manifest.get('algorithm', 'Classifier')} · {int(manifest.get('test_rows', 0)):,} held-out URLs · calibrated with a separate calibration partition")
    st.caption(f"Dataset: {manifest.get('dataset_kind', 'See release manifest')} · source: {manifest.get('dataset_source', 'See release manifest')}")
    st.caption("Metrics below evaluate the learned classifier. Independent domain-warning rules are tested separately and are not included in these benchmark figures.")
    with st.expander("All held-out metrics & data partitions"):
        st.dataframe(pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())}), width="stretch", hide_index=True)
        rows = {partition.title(): manifest.get(f"{partition}_rows", 0) for partition in ("training", "validation", "calibration", "test")}
        st.dataframe(pd.DataFrame({"Partition": list(rows), "URLs": list(rows.values())}), hide_index=True, width="stretch")
        st.caption("Registrable domains are kept separate across partitions to reduce evaluation leakage.")

    st.markdown("#### Candidate model comparison")
    st.caption("Uncalibrated candidates on the validation partition. Selection uses average precision, then phishing recall at the 50% threshold.")
    comparison = _read_artifact_table(bundle, "comparison.csv")
    if comparison is not None and not comparison.empty:
        labels = {"model": "Model", "average_precision": "Average precision", "recall": "Recall", "precision": "Precision", "f1": "F1", "roc_auc": "ROC AUC"}
        available = [key for key in labels if key in comparison]
        displayed = comparison[available].rename(columns=labels).copy()
        percentage_columns = [label for key, label in labels.items() if key != "model" and key in comparison]
        displayed[percentage_columns] = displayed[percentage_columns] * 100
        st.dataframe(
            displayed, hide_index=True, width="stretch",
            column_config={label: st.column_config.NumberColumn(label, format="%.2f%%") for label in percentage_columns},
        )
    else:
        st.info("Candidate comparison is not included in this release.")

    st.divider()
    importance_column, confusion_column = st.columns([1.15, 1], gap="large")
    with importance_column:
        st.markdown("#### Global feature importance")
        st.caption("Permutation importance of the calibrated winner on validation data. Error bars show variation across repetitions.")
        importance = _read_artifact_table(bundle, "feature_importance.csv")
        required = {"feature", "importance_mean", "importance_std"}
        if importance is not None and not importance.empty and required.issubset(importance.columns):
            st.plotly_chart(_importance_chart(importance), width="stretch", config={"displayModeBar": False}, key="importance_chart")
        else:
            st.info("Feature importance is not included in this release.")
        st.caption("Global importance describes overall model behavior; it does not explain a particular URL's score.")
    with confusion_column:
        st.markdown("#### Confusion matrix")
        _artifact_image(bundle, "confusion_matrix.png", "Calibrated model · held-out test set")
    with st.expander("Calibration & discrimination curves"):
        calibration, roc, precision_recall = st.columns(3)
        with calibration:
            _artifact_image(bundle, "calibration.png", "Probability calibration · test set")
        with roc:
            _artifact_image(bundle, "roc_curve.png", "ROC curve · test set")
        with precision_recall:
            _artifact_image(bundle, "pr_curve.png", "Precision–recall curve · test set")
    with st.expander("Model provenance & reproducibility"):
        st.text(f"Registry: {manifest.get('model_name', 'PhishShield')} · version {manifest.get('model_version', 'unknown')}")
        st.text(f"Source MLflow run: {manifest.get('run_id', 'unknown')}")
        st.text(f"Dataset SHA-256: {manifest.get('dataset_checksum', 'unknown')}")
        st.text(f"Model SHA-256: {manifest.get('model_sha256', 'unknown')}")
        st.caption("Download the manifest for thresholds, data counts, feature version, dependencies, and release provenance.")
        if manifest.get("dataset_provenance"):
            st.json(manifest["dataset_provenance"])
        st.download_button("Download release manifest", data=json.dumps(manifest, indent=2), file_name="phishshield_release_manifest.json", mime="application/json")
        st.caption("URL patterns may change over time. Results on this historical dataset do not guarantee performance on current attacks.")


def main() -> None:
    st.set_page_config(page_title="PhishShield · URL Risk Analyzer", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
    _inject_style()
    release_dir = Path(os.environ.get("PHISHSHIELD_RELEASE_DIR", str(APP_ROOT / "artifacts" / "release"))).expanduser().resolve()
    fingerprint = _release_fingerprint(release_dir)
    identity = (str(release_dir), fingerprint)
    if st.session_state.get("release_identity") != identity:
        st.session_state["release_identity"] = identity
        st.session_state.pop("prediction", None)
        st.session_state.pop("analysis_error", None)
    bundle, model_error = None, None
    try:
        bundle = _load_cached_release(str(release_dir), fingerprint)
    except (ModelLoadError, OSError) as exc:
        model_error = str(exc)
    _sidebar(bundle, model_error)
    st.markdown(
        """<div class="ps-hero"><div class="ps-eyebrow">PhishShield · URL intelligence</div><h1>Know the risk.<br><span>Before you click.</span></h1><p>Inspect a URL's structure with machine learning. Get an estimated phishing risk, recognizable warning signs, and transparent model evidence.</p></div>""",
        unsafe_allow_html=True,
    )
    analyzer, insights = st.tabs(["Analyze URL", "Model Insights"])
    with analyzer:
        _analyzer(bundle, model_error)
    with insights:
        _insights(bundle, model_error)
    st.markdown(
        '<div class="ps-footnote"><strong>Decision aid, not a guarantee.</strong> Results reflect URL patterns; the website has not been inspected. Submitted URLs are not fetched or persisted.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
