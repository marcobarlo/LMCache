#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Post GitHub commit statuses for Ascend NPU Jenkins jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

STATUS_CONTEXT = "Ascend NPU"
_GITHUB_AUTH_SCHEME = "Bearer"


def _authorization_header(github_token: str) -> dict[str, str]:
    """Build GitHub REST Authorization header from a runtime credential."""
    value = f"{_GITHUB_AUTH_SCHEME} {github_token.strip()}"
    return {"Authorization": value}


def _github_token_from_env() -> str:
    credential = os.environ.get("GITHUB_TOKEN", "").strip()
    if not credential:
        raise RuntimeError("GITHUB_TOKEN must be set in the environment")
    return credential


def post_status(
    *,
    github_token: str,
    owner: str,
    repo: str,
    sha: str,
    state: str,
    description: str,
    target_url: str | None = None,
) -> None:
    if state not in {"error", "failure", "pending", "success"}:
        raise ValueError(f"invalid state: {state}")
    url = f"https://api.github.com/repos/{owner}/{repo}/statuses/{sha}"
    body: dict[str, str] = {
        "state": state,
        "context": STATUS_CONTEXT,
        "description": description[:140],
    }
    if target_url:
        body["target_url"] = target_url
    data = json.dumps(body).encode("utf-8")
    headers = {
        **_authorization_header(github_token),
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "lmcache-ascend-jenkins",
    }
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status not in (200, 201):
            raise RuntimeError(f"unexpected status {response.status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="LMCache")
    parser.add_argument("--repo", default="LMCache")
    parser.add_argument("--sha", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--target-url", default=None)
    args = parser.parse_args()
    try:
        github_token = _github_token_from_env()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        post_status(
            github_token=github_token,
            owner=args.owner,
            repo=args.repo,
            sha=args.sha,
            state=args.state,
            description=args.description,
            target_url=args.target_url,
        )
    except urllib.error.HTTPError as exc:
        print(exc.read().decode(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
