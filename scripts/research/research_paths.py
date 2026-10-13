"""Fail-closed path configuration for private research data and outputs."""
from pathlib import Path
import os


def _inside(path, directory):
    return path == directory or directory in path.parents


def resolve_paths():
    source_value = os.environ.get('CAREACCESS_SOURCE')
    output_value = os.environ.get('CAREACCESS_RESEARCH_OUTPUT')
    if not source_value or not output_value:
        raise RuntimeError('Set CAREACCESS_SOURCE and CAREACCESS_RESEARCH_OUTPUT to paths outside the repository. Raw data and research outputs must remain private.')
    repo_value = os.environ.get('CAREACCESS_REPOSITORY')
    if repo_value:
        repository = Path(repo_value).expanduser().resolve(strict=True)
    else:
        repository = next((p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file() and (p/'careaccess').is_dir()), None)
        if repository is None:
            raise RuntimeError('Cannot infer repository. Set CAREACCESS_REPOSITORY to the CareAccess checkout root.')
    if not (repository/'careaccess').is_dir() or not (repository/'models').is_dir() or not (repository/'reports').is_dir():
        raise RuntimeError('CAREACCESS_REPOSITORY must identify a CareAccess checkout containing careaccess, models, and reports.')
    source = Path(source_value).expanduser().resolve(strict=True)
    output = Path(output_value).expanduser().resolve(strict=False)
    if not source.is_file():
        raise RuntimeError('CAREACCESS_SOURCE must be the locally authorized original CSV file.')
    if _inside(source, repository) or _inside(output, repository):
        raise RuntimeError('Refusing source or research output inside the repository. Use private external paths.')
    # Ensure artifacts cannot overwrite source, repository parents, or existing non-directory paths.
    if output == source or _inside(repository, output):
        raise RuntimeError('Research output must be a separate private directory, not a repository ancestor or source file.')
    if output.exists() and not output.is_dir():
        raise RuntimeError('CAREACCESS_RESEARCH_OUTPUT must be a directory.')
    output.mkdir(parents=True, exist_ok=True)
    # Resolve again to reject an output symlink targeting the repository.
    output = output.resolve(strict=True)
    if _inside(output, repository):
        raise RuntimeError('Research output resolves inside the repository.')
    return source, output, repository
