#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Post GitHub commit statuses for Ascend NPU Jenkins jobs."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

STATUS_CONTEXT = "Ascend NPU"


def post_status(
    *,
    token: str,
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
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "lmcache-ascend-jenkins",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status not in (200, 201):
            raise RuntimeError(f"unexpected status {response.status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=None, help="GitHub token (or GITHUB_TOKEN)")
    parser.add_argument("--owner", default="LMCache")
    parser.add_argument("--repo", default="LMCache")
    parser.add_argument("--sha", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--target-url", default=None)
    args = parser.parse_args()
    token = args.token or __import__("os").environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN or --token required", file=sys.stderr)
        return 2
    try:
        post_status(
            token=token,
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
