#!/usr/bin/env bash
# Upload the Ascend pipeline from a clean clone of an allowlisted base branch.
# The Buildkite steps editor must invoke this file from that clone.
# Do not run it from the pull request checkout: a PR can replace this file.
set -euo pipefail

TRUSTED_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CHECKOUT="${BUILDKITE_BUILD_CHECKOUT_PATH:-}"

if [[ -z "${CHECKOUT}" ]]; then
    echo "Ascend CI: BUILDKITE_BUILD_CHECKOUT_PATH is not set" >&2
    exit 1
fi

trusted_real="$(readlink -f "${TRUSTED_ROOT}")"
checkout_real="$(readlink -f "${CHECKOUT}")"
case "${trusted_real}" in
    "${checkout_real}"|"${checkout_real}"/*)
        echo "Ascend CI: refusing to run the trusted bootstrap from the job checkout" >&2
        exit 1
        ;;
esac

cd "${CHECKOUT}"
# The checkout is untrusted. Disable hooks before any git command in it.
git config --local core.hooksPath /dev/null

export LMCACHE_CI_TRUSTED_ROOT="${TRUSTED_ROOT}"
bash "${TRUSTED_ROOT}/.buildkite/k3_tests/common_scripts/upload-pipeline.sh" \
    "${TRUSTED_ROOT}/.buildkite/k3_tests/ascend/pipeline.yml"
