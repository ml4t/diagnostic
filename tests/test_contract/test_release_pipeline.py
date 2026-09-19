"""Behavioral contracts for the commit-bound release helpers."""

from __future__ import annotations

import io
import re
import tarfile
import zipfile
from copy import deepcopy
from email import policy
from email.message import Message
from pathlib import Path

import pytest

from scripts.check_release_preflight import (
    PublicationState,
    preflight_failures,
    version_failure,
)
from scripts.run_readme_quickstart import extract_quick_start
from scripts.verify_documentation_identity import identity_failures
from scripts.verify_published_release import (
    EXPECTED_AUTHOR,
    EXPECTED_CLASSIFIERS,
    EXPECTED_DESCRIPTION,
    EXPECTED_KEYWORDS,
    EXPECTED_MAINTAINER,
    EXPECTED_URLS,
    published_release_failures,
)
from scripts.verify_release_artifacts import artifact_metadata, metadata_failures
from scripts.write_release_manifest import release_manifest

COMMIT = "a" * 40
VERSION = "1.2.3"
REPOSITORY_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("version", ["1.2", "v1.2.3", "1.2.3rc1", "1.2.3+local"])
def test_release_preflight_rejects_nonstable_versions(version: str) -> None:
    assert version_failure(version) is not None


def test_release_preflight_requires_exact_unpublished_main_candidate() -> None:
    assert (
        preflight_failures(
            version=VERSION,
            source_version=VERSION,
            candidate_sha=COMMIT,
            checked_out_sha=COMMIT,
            workflow_sha=COMMIT,
            main_sha=COMMIT,
            publication=PublicationState(False, False, False),
        )
        == []
    )
    failures = preflight_failures(
        version=VERSION,
        source_version="1.2.2",
        candidate_sha=COMMIT,
        checked_out_sha="b" * 40,
        workflow_sha="c" * 40,
        main_sha="d" * 40,
        publication=PublicationState(True, True, True),
    )
    assert len(failures) == 7


def test_release_manifest_records_exact_artifact_bytes(tmp_path: Path) -> None:
    wheel = tmp_path / "ml4t_diagnostic-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ml4t_diagnostic-1.2.3.dist-info/METADATA", "Version: 1.2.3\n")
    sdist = tmp_path / "ml4t_diagnostic-1.2.3.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        contents = b"source"
        info = tarfile.TarInfo("ml4t_diagnostic-1.2.3/README.md")
        info.size = len(contents)
        archive.addfile(info, io.BytesIO(contents))

    manifest = release_manifest(tmp_path, version=VERSION, commit=COMMIT)

    assert manifest["version"] == VERSION
    assert manifest["commit"] == COMMIT
    assert set(manifest["artifacts"]) == {"wheel", "sdist"}
    assert all(len(record["sha256"]) == 64 for record in manifest["artifacts"].values())


def canonical_metadata() -> Message:
    metadata = Message()
    for name, value in {
        "Name": "ml4t-diagnostic",
        "Version": VERSION,
        "Summary": EXPECTED_DESCRIPTION,
        "Author-email": EXPECTED_AUTHOR,
        "Maintainer-email": EXPECTED_MAINTAINER,
        "License-Expression": "MIT",
        "Requires-Python": ">=3.12,<3.15",
        "Keywords": ",".join(sorted(EXPECTED_KEYWORDS)),
        "License-File": "LICENSE",
    }.items():
        metadata[name] = value
    for label, url in EXPECTED_URLS.items():
        metadata["Project-URL"] = f"{label}, {url}"
    return metadata


def test_wheel_metadata_validation_detects_identity_drift() -> None:
    metadata = canonical_metadata()
    assert metadata_failures(metadata, expected_version=VERSION) == []
    metadata.replace_header("Summary", "Different package")
    assert any(
        "Summary" in failure for failure in metadata_failures(metadata, expected_version=VERSION)
    )


def test_sdist_and_wheel_metadata_are_both_read_from_built_archives(tmp_path: Path) -> None:
    canonical = canonical_metadata().as_bytes(policy=policy.default.clone(max_line_length=0))
    sdist = tmp_path / "ml4t_diagnostic-1.2.3.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo("ml4t_diagnostic-1.2.3/PKG-INFO")
        info.size = len(canonical)
        archive.addfile(info, io.BytesIO(canonical))
    wheel = tmp_path / "ml4t_diagnostic-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ml4t_diagnostic-1.2.3.dist-info/METADATA", canonical)

    metadata = artifact_metadata(tmp_path)

    assert set(metadata) == {"source distribution", "wheel"}
    assert all(
        metadata_failures(item, expected_version=VERSION) == [] for item in metadata.values()
    )


def published_fixture() -> tuple[dict, dict, dict]:
    artifacts = {
        "wheel": {"filename": "ml4t_diagnostic-1.2.3-py3-none-any.whl", "sha256": "b" * 64},
        "sdist": {"filename": "ml4t_diagnostic-1.2.3.tar.gz", "sha256": "c" * 64},
    }
    manifest = {
        "schema_version": 1,
        "distribution": "ml4t-diagnostic",
        "version": VERSION,
        "commit": COMMIT,
        "artifacts": deepcopy(artifacts),
    }
    pypi = {
        "info": {
            "name": "ml4t-diagnostic",
            "version": VERSION,
            "summary": EXPECTED_DESCRIPTION,
            "author_email": EXPECTED_AUTHOR,
            "maintainer_email": EXPECTED_MAINTAINER,
            "license_expression": "MIT",
            "requires_python": ">=3.12,<3.15",
            "project_urls": EXPECTED_URLS,
            "keywords": ",".join(sorted(EXPECTED_KEYWORDS)),
            "classifiers": sorted(EXPECTED_CLASSIFIERS),
        },
        "urls": [
            {"filename": record["filename"], "digests": {"sha256": record["sha256"]}}
            for record in artifacts.values()
        ],
    }
    release = {
        "tag_name": f"v{VERSION}",
        "target_commitish": COMMIT,
        "assets": [
            {"name": record["filename"], "digest": f"sha256:{record['sha256']}"}
            for record in artifacts.values()
        ],
    }
    return manifest, pypi, release


def test_published_release_must_match_manifest() -> None:
    manifest, pypi, release = published_fixture()
    assert published_release_failures(manifest, pypi, release) == []
    release["assets"][0]["digest"] = "sha256:" + "d" * 64
    assert "GitHub release artifact names or digests differ from the release manifest" in (
        published_release_failures(manifest, pypi, release)
    )


def test_documentation_identity_and_quickstart_are_executable_contracts() -> None:
    html = (
        '<meta name="ml4t-library" content="diagnostic">'
        f'<meta name="ml4t-version" content="{VERSION}">'
        f'<meta name="ml4t-commit" content="{COMMIT}">'
    )
    assert (
        identity_failures(
            html,
            expected_library="diagnostic",
            expected_version=VERSION,
            expected_commit=COMMIT,
            source="index.html",
        )
        == []
    )
    assert extract_quick_start("## Quick Start\n\n```python\nvalue = 1\n```\n") == "value = 1\n"


def test_external_workflow_actions_use_full_commit_pins() -> None:
    action = re.compile(r"^\s*uses:\s*(?!\./)([^\s#]+)@([^\s#]+)", re.MULTILINE)
    failures = []
    for workflow in sorted((REPOSITORY_ROOT / ".github/workflows").glob("*.yml")):
        for name, revision in action.findall(workflow.read_text(encoding="utf-8")):
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                failures.append(f"{workflow.name}: {name}@{revision}")
    assert failures == []


def test_release_workflow_allows_reusable_jobs_to_read_checkout() -> None:
    release_workflow = (REPOSITORY_ROOT / ".github/workflows/release.yml").read_text(
        encoding="utf-8"
    )
    workflow_permissions = release_workflow.split("\nconcurrency:", maxsplit=1)[0]
    assert re.search(r"^permissions:\n  contents: read$", workflow_permissions, re.MULTILINE)


def test_only_release_workflow_can_deploy_public_documentation() -> None:
    deploy_action = "cpina/github-action-push-to-another-repository@"
    deployers = [
        workflow.name
        for workflow in sorted((REPOSITORY_ROOT / ".github/workflows").glob("*.yml"))
        if deploy_action in workflow.read_text(encoding="utf-8")
    ]
    assert deployers == ["release.yml"]
