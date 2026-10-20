"""About, provenance and safe-use page."""

import streamlit as st

from careaccess.ui.common import (
    HOST_INPUT_NOTICE,
)


def render_about(bundle):
    report = bundle["report"]
    contract = bundle["feature_contract"]
    manifest = bundle["manifest"]
    st.header("About CareAccess ML")
    st.write(
        "An appointment prediction and outreach-planning workbench using Python, scikit-learn, PyTorch and Streamlit. Saved trained models run directly in the application; no generative-AI API supplies the probabilities."
    )
    st.link_button("View GitHub repository", "https://github.com/ibrahimbaabdullah-jpg/CareAccess-ML")
    st.link_button("View original dataset", "https://data.mendeley.com/datasets/wm6w2fvkfj/1")
    source_info = report.get("source", {})
    st.write(
        f"Training source: {source_info.get('title', 'See model report')} — {source_info.get('author', '')}. License: {source_info.get('license', 'See source')}. DOI: {source_info.get('doi', '')}."
    )
    st.caption(f"Model bundle created: {manifest.get('created_utc', 'See manifest')}")
    st.subheader("Data, model release and provenance")
    st.json(manifest, expanded=False)
    st.subheader("Input contract")
    st.json(contract, expanded=False)
    st.subheader("Privacy and safe use")
    st.warning(HOST_INPUT_NOTICE)
    st.link_button("Streamlit hosting restrictions · Community Cloud Addendum 4", "https://streamlit.io/terms-of-use")
    st.write(
        "Input is held in the active session for prediction and download; it is not added to model training or deliberately written to a database. Hosting providers may retain ordinary technical access logs. A separate local or institutional deployment still requires privacy, security, legal and clinical review; local execution alone does not establish compliance. The confirmation checkbox is an acknowledgement, not permission to bypass these restrictions."
    )
    st.write(
        "This is not a clinically approved tool. It must not be used to deny care, cancel appointments, punish patients, or make autonomous clinical decisions. A score does not establish that a person will miss a visit."
    )
    st.write(
        "Training data source, license, cleaning decisions, model cards, environment versions and reproducibility instructions are documented in the repository. Third-party data remains subject to its source license."
    )
