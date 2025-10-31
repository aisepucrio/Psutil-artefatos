# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from codecarbon import EmissionsTracker
import psutil
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import click
import requests

REPO_ROOT_DIR = Path(__file__).parent.parent.parent
assert REPO_ROOT_DIR.is_dir() and (REPO_ROOT_DIR / ".git").is_dir()


IGNORE_PREFIX_PATTERNS = [
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "python/.venv",
    "python/dist",
    "website/node_modules",
    "website/dist",
    "js/node_modules",
    "js/dist",
]

URLS_ALLOWLIST_PREFIXES = [
    "https://api.securityscorecards.dev/projects/github.com/google/magika/badge",
    "https://crates.io/crates/magika",
    "https://crates.io/crates/magika-cli",
    "https://www.unrealengine.com/en-US",
    "https://www.unrealengine.com/marketplace/en-US/store",
    "https://www.virustotal.com/",
]


@dataclass(kw_only=True)
class UriInfo:
    uri: str
    is_external: bool
    is_valid: bool
    is_pure_anchor: bool
    is_insecure: bool


@click.command()
@click.option("--skip-external-validity-check", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
def main(skip_external_validity_check: bool, verbose: bool) -> None:
    with_errors = False

    if not check_versions_are_up_to_date():
        with_errors = True

    if not check_markdown_links(skip_external_validity_check, verbose):
        with_errors = True

    if with_errors:
        print("There was at least one error.")
        sys.exit(1)

    print("Everything looks good.")


def check_versions_are_up_to_date() -> bool:
    """Checks that the mentioned latest versions and models are up to date.
    Returns True if everything is good, False otherwise."""

    python_latest_stable_version = get_python_latest_stable_version()
    python_default_model_name = get_python_default_model_name()
    rust_default_model_name = get_rust_default_model_name()

    print(
        f"INFO: {python_latest_stable_version=} {python_default_model_name=} {rust_default_model_name=}"
    )

    expected_lines = [
        f"> - The documentation on GitHub refers to the latest, potentially unreleased and unstable version of Magika. The latest stable release of the `magika` Python package is `{python_latest_stable_version}`, and you can consult the associated documentation [here](https://github.com/google/magika/blob/python-v{python_latest_stable_version}/python/README.md). You can install the latest stable version with: `pip install magika`.",
        f"- Trained and evaluated on a dataset of ~100M files across [200+ content types](./assets/models/{python_default_model_name}/README.md).",
        f"- [List of supported content types by the latest model, `{python_default_model_name}`](./assets/models/{python_default_model_name}/README.md)",
    ]

    readme_content_lines_set = set(
        (REPO_ROOT_DIR / "README.md").read_text().split("\n")
    )

    with_errors = False
    for expected_line in expected_lines:
        if expected_line not in readme_content_lines_set:
            print(f'ERROR: could not find the following line: "{expected_line}"')
            with_errors = True

    return not with_errors


def get_python_latest_stable_version() -> str:
    res = requests.get("https://pypi.org/pypi/magika/json")
    assert res.status_code == 200
    latest_stable_version = res.json().get("info", {}).get("version", None)
    assert latest_stable_version is not None
    return latest_stable_version


def get_python_default_model_name() -> str:
    default_model_name = None
    magika_path = REPO_ROOT_DIR / "python" / "src" / "magika" / "magika.py"
    assert magika_path.is_file()
    for line in magika_path.read_text().split("\n"):
        m = re.fullmatch('_DEFAULT_MODEL_NAME = "([a-zA-Z0-9_]+)"', line)
        if m:
            assert default_model_name is None
            default_model_name = m.group(1)

    return default_model_name


def get_rust_default_model_name() -> str:
    model_symlink_path = REPO_ROOT_DIR / "rust" / "gen" / "model"
    assert model_symlink_path.is_symlink()
    return model_symlink_path.readlink().name


def check_markdown_links(skip_external_validity_check: bool, verbose: bool) -> bool:
    """Checks that links in Markdown files are OK. Returns True if everything is
    good, False otherwise."""
    with_errors = False
    for path in enumerate_markdown_files_in_dir(Path(".")):
        if verbose:
            print(f"Analyzing {path}")
        for ui in extract_uris_infos_from_file(
            path,
            skip_external_validity_check=skip_external_validity_check,
            verbose=verbose,
        ):
            if not ui.is_valid:
                with_errors = True
                print(
                    f"ERROR: {path.relative_to(REPO_ROOT_DIR)} has non-valid uri: {ui.uri}"
                )
            if str(path.relative_to(REPO_ROOT_DIR)) == "python/README.md":
                if not ui.is_external and not ui.is_pure_anchor:
                    with_errors = True
                    print(
                        f"ERROR: {path.relative_to(REPO_ROOT_DIR)}, in python/, has a non-external uri: {ui.uri}"
                    )
            if str(path.relative_to(REPO_ROOT_DIR)) == "js/README.md":
                if not ui.is_external and not ui.is_pure_anchor:
                    with_errors = True
                    print(
                        f"ERROR: {path.relative_to(REPO_ROOT_DIR)}, in python/, has a non-external uri: {ui.uri}"
                    )

    return not with_errors


def enumerate_markdown_files_in_dir(rel_dir: Path) -> list[Path]:
    if rel_dir.is_absolute():
        print(f"{rel_dir} is not relative")
        sys.exit(1)
    a_dir = REPO_ROOT_DIR / rel_dir
    assert a_dir.is_dir()
    paths: list[Path] = []
    for path in sorted(a_dir.rglob("*.md")):
        if not should_ignore_file(path):
            paths.append(path)
    return paths


def should_ignore_file(path: Path) -> bool:
    for exclude_prefix_pattern in IGNORE_PREFIX_PATTERNS:
        if str(path.relative_to(REPO_ROOT_DIR)).startswith(exclude_prefix_pattern):
            return True
    return False


def extract_uris_infos_from_file(
    path: Path, skip_external_validity_check: bool, verbose: bool
) -> list[UriInfo]:
    uris = find_uris_in_file(path)
    uris_infos = []
    for uri in uris:
        if verbose:
            print(f"Analyzing uri: {uri}")
        uri_info = analyze_uri(path, uri, skip_external_validity_check)
        uris_infos.append(uri_info)
    return uris_infos


def find_uris_in_file(path: Path) -> list[str]:
    uri_regex = r"\[.*?\]\((.*?)\)"
    return re.findall(uri_regex, path.read_text())


def analyze_uri(
    path: Path, uri: str, skip_external_validity_check: bool
) -> UriInfo:
    is_external = uri.startswith("http://") or uri.startswith("https://")
    is_valid = None
    is_pure_anchor = None
    is_insecure = None

    if is_external:
        is_pure_anchor = False
        is_insecure = uri.startswith("http://")
        if is_insecure:
            print(f"WARNING: {uri} is not using https")

        if skip_external_validity_check:
            print(f"WARNING: skipping check for {uri}")
            is_valid = True
        else:
            is_valid = check_external_uri(uri)
    else:
        is_insecure = False
        if uri.startswith("#"):
            is_valid = True
            is_pure_anchor = True
        else:
            is_pure_anchor = False
            is_valid = check_internal_uri(path, uri)

    assert is_valid is not None
    assert is_pure_anchor is not None
    assert is_insecure is not None

    return UriInfo(
        uri=uri,
        is_external=is_external,
        is_valid=is_valid,
        is_pure_anchor=is_pure_anchor,
        is_insecure=is_insecure,
    )


def check_external_uri(uri: str) -> bool:
    for url_allowlist_prefix in URLS_ALLOWLIST_PREFIXES:
        if uri.startswith(url_allowlist_prefix):
            return True
    try:
        r = requests.head(uri, allow_redirects=True, timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not validate {uri}: {e}")
        return False


def check_internal_uri(path: Path, uri: str) -> bool:
    if Path(uri).is_absolute():
        return False
    if "#" in uri:
        rel_file_path = uri.split("#")[0]
    else:
        rel_file_path = uri
    abs_path = path.parent / rel_file_path
    return abs_path.is_file() or abs_path.is_dir()


if __name__ == "__main__":
    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\magika\python\scripts\check_documentation.py")
    tracker.start()

    mem_start = psutil.virtual_memory().used / (1024**2)
    cpu_start = psutil.cpu_percent(interval=None)

    main()

    mem_end = psutil.virtual_memory().used / (1024**2)
    cpu_end = psutil.cpu_percent(interval=None)

    csv_file = "psutil_data.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])
        writer.writerow([
            __file__,
            f"{mem_start:.2f}",
            f"{mem_end:.2f}",
            f"{mem_end - mem_start:.2f}",
            f"{cpu_start:.2f}",
            f"{cpu_end:.2f}"
        ])

    tracker.stop()
