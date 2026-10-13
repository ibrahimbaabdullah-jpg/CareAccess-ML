"""Research-display and host-input wording checks; no source records required."""
from pathlib import Path
from hashlib import sha256

from streamlit.testing.v1 import AppTest
from careaccess.inference import load_bundle, model_names
from careaccess.research import SUMMARY_PATH

ROOT = Path(__file__).resolve().parents[1]


def visible_text(at):
    return "\n".join(
        str(item.value)
        for kind in ["markdown", "caption", "warning", "info"]
        for item in at.get(kind)
    )


def test_home_distinguishes_research_display_from_submitted_inputs():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception and not at.error
    text = visible_text(at)
    assert "real trained models and aggregate results" in text
    assert "publicly released appointment research datasets" in text
    assert "Underlying patient records and generated appointment cohorts are not published" in text
    assert "Input warnings concern data submitted to this host" in text
    assert "do not classify all displayed research aggregates or model outputs as sensitive" in text
    assert "Submitted inputs must be permitted, nonsensitive and non-identifying" in text


def test_research_notice_is_scoped_to_streamlit_main_release():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Research / Vitória").run()
    assert not at.exception and not at.error
    text = visible_text(at)
    assert "aggregate-only, main-branch Streamlit release" in text
    assert "Separate branches or browser-demo releases have their own asset-specific notices" in text
    assert "no Vitória source records or research weights" in text


def test_docs_keep_input_boundary_without_blanket_source_classification():
    for name in ["README.md", "docs/privacy-security.md", "docs/retraining.md", "docs/data-audit.md"]:
        text = (ROOT / name).read_text()
        assert "sensitive quasi-identifiers" not in text
        assert "birth date" in text.lower() or "date of birth" in text.lower()
    for name in ["README.md", "docs/privacy-security.md"]:
        text = (ROOT / name).read_text()
        assert "Streamlit Community Cloud Addendum 4" in text
        assert "checking a consent box does not establish" in text
        assert "Source-data evaluation is performed offline" in text


def test_separate_browser_demo_links_preserve_original_models_and_summary():
    url = "https://ibrahimbaabdullah-jpg.github.io/CareAccess-ML/"
    expected_models = set(model_names(load_bundle()))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    for page in ["Home / Dashboard", "Research / Vitória"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception and not at.error
        links = [item for item in at.get("link_button") if item.proto.url == url]
        assert len(links) == 1
        assert links[0].proto.label == "Open separate browser-only research demo"
        text = visible_text(at)
        assert "frozen Vitória XGBoost" in text
        assert "does not replace or validate the original Mendeley models" in text
    at.sidebar.radio[0].set_value("Live Prediction").run()
    assert not at.exception and not at.error
    assert set(at.selectbox(key="live_model").options) == expected_models
    assert not any("XGBoost" in name or "Vitória" in name for name in expected_models)
    assert at.date_input[0].value is None and at.time_input[0].value is None
    assert not at.checkbox(key="live_confirm").value
    assert "last_single_prediction" not in at.session_state
    assert sha256(SUMMARY_PATH.read_bytes()).hexdigest() == "316d10de679e08019458befb07464b91c18139155db7120fa36e382f99f041dc"
    assert url in (ROOT / "README.md").read_text()
