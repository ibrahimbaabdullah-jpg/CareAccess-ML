"""Fail closed before loading any artifact from a damaged manifest."""
import json
from pathlib import Path

import pytest

from careaccess import inference

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest():
    return json.loads((ROOT / 'models/manifest.json').read_text())


def test_configuration_provenance_is_recorded(manifest):
    # Training-source hashes are provenance; artifact checksums gate loading.
    for filename, digest in manifest['configuration_sha256'].items():
        assert (ROOT / filename).is_file()
        assert len(digest) == 64 and int(digest, 16) >= 0


@pytest.mark.parametrize('artifact', ['preprocessor.skops', *inference.MODEL_FILES.values()])
def test_every_loaded_artifact_requires_checksum(manifest, artifact):
    del manifest['sha256'][artifact]
    with pytest.raises(ValueError, match='Every required'):
        inference.verify_artifacts(ROOT / 'models', manifest)


@pytest.mark.parametrize('filename', ['../logistic.skops', '/tmp/logistic.skops', 'unchecked.skops'])
def test_manifest_cannot_redirect_loaded_models(manifest, filename):
    manifest['model_files']['Logistic Regression'] = filename
    with pytest.raises(ValueError, match='Unexpected bundled'):
        inference.verify_artifacts(ROOT / 'models', manifest)


def test_extra_checksummed_file_rejected(manifest):
    manifest['sha256']['extra.skops'] = '0' * 64
    with pytest.raises(ValueError, match='Every required'):
        inference.verify_artifacts(ROOT / 'models', manifest)


def test_checksum_mismatch_rejected(manifest):
    manifest['sha256']['preprocessor.skops'] = '0' * 64
    with pytest.raises(ValueError, match='checksum mismatch'):
        inference.verify_artifacts(ROOT / 'models', manifest)


def test_invalid_digest_rejected(manifest):
    manifest['sha256']['preprocessor.skops'] = None
    with pytest.raises(ValueError, match='Invalid model checksum'):
        inference.verify_artifacts(ROOT / 'models', manifest)


def test_manifest_cannot_expand_type_allowlist(manifest):
    manifest['skops_types']['logistic.skops'] = ['sklearn.unreviewed.Type']
    with pytest.raises(ValueError, match='type allowlist'):
        inference.verify_artifacts(ROOT / 'models', manifest)


def test_artifact_cannot_be_symlink_outside_bundle(manifest, tmp_path):
    (tmp_path / 'preprocessor.skops').symlink_to(ROOT / 'models/preprocessor.skops')
    with pytest.raises(ValueError, match='location'):
        inference.verify_artifacts(tmp_path, manifest)


def test_incomplete_manifest_stops_before_deserialization(manifest, tmp_path, monkeypatch):
    del manifest['sha256']['logistic.skops']
    models = tmp_path / 'models'
    models.mkdir()
    (models / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(inference, 'ROOT', tmp_path)
    def must_not_load(*args, **kwargs):
        pytest.fail('Deserializer called before complete artifact verification.')
    monkeypatch.setattr(inference.sio, 'get_untrusted_types', must_not_load)
    monkeypatch.setattr(inference.sio, 'load', must_not_load)
    monkeypatch.setattr(inference, 'load_file', must_not_load)
    with pytest.raises(ValueError, match='Every required'):
        inference.load_bundle()


def test_artifact_types_cannot_exceed_code_owned_allowlist(monkeypatch):
    monkeypatch.setattr(inference.sio, 'get_untrusted_types', lambda **kwargs: ['sklearn.unreviewed.Type'])
    def must_not_load(*args, **kwargs):
        pytest.fail('Unreviewed deserialization type was loaded.')
    monkeypatch.setattr(inference.sio, 'load', must_not_load)
    with pytest.raises(ValueError, match='Unexpected model types'):
        inference.load_bundle()
