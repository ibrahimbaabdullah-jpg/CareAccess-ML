"""Research-display and host-input wording checks; no source records required."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

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
