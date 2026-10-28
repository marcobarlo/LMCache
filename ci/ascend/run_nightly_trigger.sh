#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Resolve latest origin/dev and trigger the nightly Ascend Jenkins job.
set -euo pipefail

JENKINS_URL="${JENKINS_URL:?set JENKINS_URL}"
JENKINS_JOB="${JENKINS_JOB:-lmcache-ascend-nightly}"
JENKINS_USER="${JENKINS_USER:?set JENKINS_USER}"
JENKINS_TOKEN="${JENKINS_TOKEN:?set JENKINS_TOKEN}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

cd "$REPO_ROOT"
git fetch origin dev
GIT_SHA="$(git rev-parse origin/dev)"

PATHS="$(tr '\n' ' ' < ci/ascend/nightly_pytest_paths.txt | xargs)"

PARAMS="$(python3 - <<PY
import urllib.parse
print(urllib.parse.urlencode({
    "MODE": "nightly",
    "GIT_SHA": "${GIT_SHA}",
    "PYTEST_PATHS": """${PATHS}""".strip(),
    "AGENT_LABEL": "ascend && arm64 && 910b",
    "PR_NUMBER": "0",
}))
PY
)"

curl -sf -X POST \
  -u "${JENKINS_USER}:${JENKINS_TOKEN}" \
  "${JENKINS_URL%/}/job/${JENKINS_JOB}/buildWithParameters?${PARAMS}"

echo "Queued nightly ${GIT_SHA}"
