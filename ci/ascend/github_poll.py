#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Poll GitHub for gated LMCache PRs and trigger the Ascend Jenkins job."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from github_status import post_status  # noqa: E402

NPU_LABEL = "npu"
BASE_BRANCHES = ("dev",)


def _request(
    token: str,
    method: str,
    url: str,
    body: dict | None = None,
) -> object:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "lmcache-ascend-jenkins-poll",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_open_prs(token: str, owner: str, repo: str) -> list[dict]:
    pulls: list[dict] = []
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {"state": "open", "per_page": "100", "page": str(page)}
        )
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls?{query}"
        batch = _request(token, "GET", url)
        if not isinstance(batch, list) or not batch:
            break
        pulls.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return pulls


def pr_has_npu_label(pr: dict) -> bool:
    return any(label.get("name") == NPU_LABEL for label in pr.get("labels", []))


def pr_has_write_approval(token: str, owner: str, repo: str, pr_number: int) -> bool:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    reviews = _request(token, "GET", url)
    if not isinstance(reviews, list):
        return False
    for review in reviews:
        if review.get("state") != "APPROVED":
            continue
        assoc = review.get("author_association", "")
        if assoc in {"OWNER", "MEMBER", "COLLABORATOR"}:
            return True
    return False


def pr_is_eligible(token: str, owner: str, repo: str, pr: dict) -> bool:
    base = pr.get("base", {}).get("ref", "")
    if not (base in BASE_BRANCHES or base.startswith("release-")):
        return False
    if pr_has_npu_label(pr):
        return True
    return pr_has_write_approval(token, owner, repo, int(pr["number"]))


def compute_pytest_paths(repo_root: Path, base_sha: str, head_sha: str) -> str:
    script = repo_root / "ci" / "ascend" / "select_tests.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--base",
            base_sha,
            "--head",
            head_sha,
            "--format",
            "shell",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return result.stdout.strip()


def trigger_jenkins(
    *,
    jenkins_url: str,
    job: str,
    user: str,
    token: str,
    git_sha: str,
    pytest_paths: str,
    pr_number: int,
) -> str:
    params = urllib.parse.urlencode(
        {
            "MODE": "pr",
            "GIT_SHA": git_sha,
            "PYTEST_PATHS": pytest_paths,
            "AGENT_LABEL": "ascend && arm64 && 910b",
            "PR_NUMBER": str(pr_number),
        }
    )
    build_url = (
        f"{jenkins_url.rstrip('/')}/job/{urllib.parse.quote(job)}"
        f"/buildWithParameters?{params}"
    )
    auth = base64.b64encode(f"{user}:{token}".encode()).decode()
    req = urllib.request.Request(
        build_url,
        method="POST",
        headers={
            "Authorization": f"Basic {auth}",
            "User-Agent": "lmcache-ascend-jenkins-poll",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        location = resp.headers.get("Location", "")
    return location


def load_state(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="LMCache")
    parser.add_argument("--repo", default="LMCache")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--jenkins-url", required=True)
    parser.add_argument("--jenkins-job", default="lmcache-ascend-pr")
    parser.add_argument("--jenkins-user", default=os.environ.get("JENKINS_USER"))
    parser.add_argument("--jenkins-token", default=os.environ.get("JENKINS_TOKEN"))
    parser.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("/var/lib/jenkins/ascend-pr-state.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.github_token:
        print("GITHUB_TOKEN required", file=sys.stderr)
        return 2
    if not args.dry_run and (not args.jenkins_user or not args.jenkins_token):
        print("JENKINS_USER and JENKINS_TOKEN required", file=sys.stderr)
        return 2

    repo_root = args.repo_root.resolve()
    state = load_state(args.state_file)
    pulls = list_open_prs(args.github_token, args.owner, args.repo)

    for pr in pulls:
        number = int(pr["number"])
        head_sha = pr["head"]["sha"]
        base_sha = pr["base"]["sha"]
        key = f"pr-{number}"
        if state.get(key) == head_sha:
            continue
        if not pr_is_eligible(args.github_token, args.owner, args.repo, pr):
            continue

        for ref in (base_sha, head_sha):
            subprocess.run(
                ["git", "-C", str(repo_root), "fetch", "--depth=1", "origin", ref],
                check=False,
                capture_output=True,
            )

        try:
            paths = compute_pytest_paths(repo_root, base_sha, head_sha)
        except subprocess.CalledProcessError as exc:
            print(f"PR #{number}: select_tests failed: {exc.stderr}", file=sys.stderr)
            continue
        if not paths:
            post_status(
                token=args.github_token,
                owner=args.owner,
                repo=args.repo,
                sha=head_sha,
                state="success",
                description="No Ascend test paths for this diff",
            )
            state[key] = head_sha
            continue

        if args.dry_run:
            print(f"Would build PR #{number} {head_sha} paths={paths}")
            continue

        post_status(
            token=args.github_token,
            owner=args.owner,
            repo=args.repo,
            sha=head_sha,
            state="pending",
            description="Ascend NPU tests queued",
        )
        try:
            queue_url = trigger_jenkins(
                jenkins_url=args.jenkins_url,
                job=args.jenkins_job,
                user=args.jenkins_user,
                token=args.jenkins_token,
                git_sha=head_sha,
                pytest_paths=paths,
                pr_number=number,
            )
            print(f"Queued PR #{number} {head_sha} -> {queue_url}")
            state[key] = head_sha
        except urllib.error.HTTPError as err:
            print(err.read().decode(), file=sys.stderr)
            post_status(
                token=args.github_token,
                owner=args.owner,
                repo=args.repo,
                sha=head_sha,
                state="error",
                description="Failed to queue Ascend Jenkins job",
            )

    if not args.dry_run:
        save_state(args.state_file, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
