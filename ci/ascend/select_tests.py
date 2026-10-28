#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Map a git diff to pytest paths for Ascend PR jobs, or signal the nightly suite."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

NIGHTLY_SENTINEL = "__NIGHTLY__"

BUILD_NATIVE_PREFIXES = (
    "csrc/",
    "setup.py",
    "setup_extensions/",
    "pyproject.toml",
    "CMakeLists.txt",
    "rust/",
)

CUDA_MODULE_RE = re.compile(
    r"^pytestmark\s*=.*pytest\.mark\.cuda",
    re.MULTILINE,
)


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def changed_files(repo: Path, base: str, head: str) -> list[str]:
    out = _run_git(repo, "diff", "--name-only", f"{base}...{head}")
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_cuda_only_test(repo: Path, test_path: str) -> bool:
    path = repo / test_path
    if not path.is_file():
        return False
    head = path.read_text(encoding="utf-8", errors="replace")[:8192]
    if CUDA_MODULE_RE.search(head):
        return True
    if re.search(r"^pytestmark\s*=\s*\[.*pytest\.mark\.cuda", head, re.MULTILINE):
        return True
    return False


def needs_nightly(paths: list[str]) -> bool:
    for path in paths:
        for prefix in BUILD_NATIVE_PREFIXES:
            if path == prefix or path.startswith(prefix):
                return True
    return False


def find_test_for_stem(repo: Path, stem: str) -> Path | None:
    matches = sorted(repo.glob(f"tests/**/test_{stem}.py"))
    if not matches:
        return None
    return matches[0]


def map_lmcache_change(repo: Path, path: str) -> str | None:
    if not path.startswith("lmcache/"):
        return None
    stem = Path(path).stem
    hit = find_test_for_stem(repo, stem)
    if hit is not None:
        return str(hit.relative_to(repo))
    if path.startswith("lmcache/v1/multiprocess/"):
        return "tests/v1/multiprocess"
    if "storage_backend/p2p_backend" in path:
        return "tests/v1/storage_backend/test_p2p_backend_with_controller.py"
    if "distributed/l2_adapters/p2p_l2_adapter" in path:
        return "tests/v1/distributed/l2_adapters/test_p2p_l2_adapter.py"
    if path.startswith("lmcache/v1/distributed/"):
        return "tests/v1/distributed/l2_adapters/test_p2p_l2_adapter.py"
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "lmcache" and parts[1] == "v1":
        package = parts[2]
        candidate = repo / "tests" / "v1" / package
        if candidate.is_dir():
            return str(candidate.relative_to(repo))
    return None


def select_paths(repo: Path, base: str, head: str) -> list[str]:
    paths = changed_files(repo, base, head)
    if not paths:
        return []
    if needs_nightly(paths):
        return [NIGHTLY_SENTINEL]

    selected: set[str] = set()
    for path in paths:
        if path.startswith("tests/") and path.endswith(".py"):
            if is_cuda_only_test(repo, path):
                continue
            selected.add(path)
            continue
        if path.startswith("lmcache/"):
            mapped = map_lmcache_change(repo, path)
            if mapped:
                selected.add(mapped)
            continue
        if path.startswith(".github/workflows/") and "multiprocess" in path:
            selected.add(NIGHTLY_SENTINEL)

    if NIGHTLY_SENTINEL in selected:
        return [NIGHTLY_SENTINEL]
    return sorted(selected)


def load_nightly_paths(repo: Path) -> list[str]:
    manifest = repo / "ci" / "ascend" / "nightly_pytest_paths.txt"
    lines: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True, help="Merge base ref, e.g. origin/dev")
    parser.add_argument("--head", required=True, help="PR head SHA or ref")
    parser.add_argument(
        "--format",
        choices=("paths", "shell"),
        default="shell",
        help="Print space-separated paths for Jenkins (default) or one path per line",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()

    picked = select_paths(repo, args.base, args.head)
    if not picked:
        return 0
    if picked == [NIGHTLY_SENTINEL]:
        picked = load_nightly_paths(repo)

    if args.format == "paths":
        for item in picked:
            print(item)
    else:
        print(" ".join(picked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
