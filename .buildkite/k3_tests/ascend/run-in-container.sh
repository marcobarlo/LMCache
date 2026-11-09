#!/usr/bin/env bash
# Run Ascend NPU pytest in the vLLM-Ascend container on a self-hosted agent.
#
# On Buildkite this file must be the copy from an allowlisted base branch,
# executed from outside the job checkout. The checkout is test input only.
set -euo pipefail
set +x

if [[ -n "${BUILDKITE_BUILD_CHECKOUT_PATH:-}" ]]; then
    script_path="$(readlink -f "$0")"
    checkout_path="$(readlink -f "${BUILDKITE_BUILD_CHECKOUT_PATH}")"
    case "${script_path}" in
        "${checkout_path}"|"${checkout_path}"/*)
            echo "Ascend CI: refusing to execute the host runner from the job checkout" >&2
            exit 1
            ;;
    esac
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${BUILDKITE_BUILD_CHECKOUT_PATH:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
IMAGE="${ASCEND_CI_IMAGE:-quay.io/ascend/vllm-ascend:v0.26.0rc1}"
SOC_VERSION="${SOC_VERSION:-Ascend910B4}"
ASCEND_PLUGIN_REF="${LMCACHE_ASCEND_REF:-main}"
ASCEND_PLUGIN_REPO="${LMCACHE_ASCEND_REPO:-git@github.com:LMCache/LMCache-Ascend.git}"
ASCEND_PLUGIN_ROOT=""
ARTIFACT_DIR="${REPO_ROOT}/ascend-ci-artifacts"
STAGE_DIR=""
INNER=""
ENV_FILE=""

log() {
    echo "--- :ascend: $*"
}

redact_addresses() {
    # Keep the npu-smi health table, including PCI bus ids. Strip network
    # addresses if a future driver build prints them.
    sed -E \
        -e 's/\b([0-9]{1,3}\.){3}[0-9]{1,3}\b/[redacted-ip]/g' \
        -e 's/\b([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b/[redacted-mac]/g' \
        -e 's/\b([0-9A-Fa-f]{0,4}:){3,7}[0-9A-Fa-f]{0,4}\b/[redacted-ip]/g' \
        -e 's/(^|[^0-9A-Fa-f:])::[0-9A-Fa-f:.]*/\1[redacted-ip]/g'
}

fail() {
    local msg
    msg="$(printf '%s' "$*" | redact_addresses)"
    mkdir -p "${ARTIFACT_DIR}" 2>/dev/null || true
    printf '[ascend-ci] ERROR: %s\n' "${msg}" \
        | tee -a "${ARTIFACT_DIR}/failure.log" >&2
    exit 1
}

cleanup() {
    local status=$?
    if [[ -n "${STAGE_DIR}" && -d "${STAGE_DIR}" ]]; then
        rm -rf "${STAGE_DIR}" 2>/dev/null || true
    fi
    if [[ -n "${INNER}" && -f "${INNER}" ]]; then
        rm -f "${INNER}" 2>/dev/null || true
    fi
    if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
        rm -f "${ENV_FILE}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT

validate_token() {
    local label="$1"
    local value="$2"
    [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._+/-]{0,160}$ ]] \
        || fail "${label} contains unsupported characters"
    case "${value}" in
        *..*) fail "${label} contains unsupported characters" ;;
    esac
}

validate_proxy() {
    local value="$1"
    [[ -z "${value}" ]] && return 0
    case "${value}" in
        http://*|https://*) ;;
        *) fail "HTTP proxy URL is invalid" ;;
    esac
    case "${value}" in
        *[[:space:]]*|*[\'\"\`\$\;\&\|\<\>]* )
            fail "HTTP proxy URL is invalid"
            ;;
    esac
}

on_buildkite() {
    [[ -n "${BUILDKITE_BUILD_ID:-}" || -n "${BUILDKITE_AGENT_NAME:-}" ]]
}

mkdir -p "${ARTIFACT_DIR}/junit"
chmod 1777 "${ARTIFACT_DIR}/junit" || true
command -v docker >/dev/null 2>&1 || fail "docker is required on the Ascend agent"
command -v git >/dev/null 2>&1 || fail "git is required to prepare LMCache-Ascend"
command -v tar >/dev/null 2>&1 || fail "tar is required to stage the test sources"

validate_token "SOC_VERSION" "${SOC_VERSION}"
[[ "${IMAGE}" =~ ^[A-Za-z0-9][A-Za-z0-9._:/-]{0,200}$ ]] \
    || fail "container image reference is invalid"
validate_token "LMCACHE_ASCEND_REF" "${ASCEND_PLUGIN_REF}"
[[ "${ASCEND_PLUGIN_REPO}" =~ ^(git@github\.com:|https://github\.com/)[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?$ ]] \
    || fail "LMCache-Ascend repository URL is not an allowed GitHub remote"
validate_proxy "${ASCEND_CI_HTTP_PROXY:-}"
validate_proxy "${ASCEND_CI_HTTPS_PROXY:-}"

if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>&1 | redact_addresses | tee "${ARTIFACT_DIR}/host-npu-smi.txt" || true
fi

prepare_ascend_plugin_checkout() {
    local root
    if [[ -n "${LMCACHE_ASCEND_CHECKOUT_PATH:-}" ]]; then
        root="${LMCACHE_ASCEND_CHECKOUT_PATH}"
        case "${root}" in
            /*) ;;
            *) fail "plugin checkout path must be absolute" ;;
        esac
        case "${root}" in
            "${REPO_ROOT}"|"${REPO_ROOT}"/*|"${ARTIFACT_DIR}"|"${ARTIFACT_DIR}"/*)
                fail "plugin checkout must not live inside the job checkout or artifact directory"
                ;;
        esac
        [[ -d "${root}/.git" ]] \
            || fail "pre-provisioned LMCache-Ascend checkout is missing"
        ASCEND_PLUGIN_ROOT="${root}"
        log "Using pre-provisioned LMCache-Ascend checkout"
        return 0
    fi

    root="${XDG_CACHE_HOME:-${HOME}/.cache}/lmcache-ascend-ci"
    case "${root}" in
        "${REPO_ROOT}"|"${REPO_ROOT}"/*|"${ARTIFACT_DIR}"|"${ARTIFACT_DIR}"/*)
            fail "plugin cache resolved inside the job checkout"
            ;;
    esac
    ASCEND_PLUGIN_ROOT="${root}"
    if [[ ! -d "${ASCEND_PLUGIN_ROOT}/.git" ]]; then
        log "Cloning LMCache-Ascend into the agent cache"
        mkdir -p "$(dirname "${ASCEND_PLUGIN_ROOT}")"
        git -c core.hooksPath=/dev/null clone --branch "${ASCEND_PLUGIN_REF}" --depth 1 \
            "${ASCEND_PLUGIN_REPO}" "${ASCEND_PLUGIN_ROOT}" \
            || fail "git clone LMCache-Ascend failed"
    else
        log "Updating cached LMCache-Ascend checkout"
        git -c core.hooksPath=/dev/null -C "${ASCEND_PLUGIN_ROOT}" fetch --depth 1 origin "${ASCEND_PLUGIN_REF}" \
            || fail "git fetch LMCache-Ascend failed"
        git -c core.hooksPath=/dev/null -C "${ASCEND_PLUGIN_ROOT}" checkout -B ci-ascend-plugin "origin/${ASCEND_PLUGIN_REF}" \
            || fail "git checkout LMCache-Ascend failed"
    fi
    git -c core.hooksPath=/dev/null -C "${ASCEND_PLUGIN_ROOT}" submodule update --init --recursive --depth 1 \
        || git -c core.hooksPath=/dev/null -C "${ASCEND_PLUGIN_ROOT}" submodule update --init --recursive \
        || fail "LMCache-Ascend submodule init failed"
}

copy_tree_without_vcs() {
    local src="$1"
    local dest="$2"
    mkdir -p "${dest}"
    tar -C "${src}" \
        --exclude=.git \
        --exclude=ascend-ci-artifacts \
        -cf - . | tar -C "${dest}" -xf -
}

stage_lmcache_sources() {
    mkdir -p "${STAGE_DIR}/LMCache"
    if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "${REPO_ROOT}" config --local core.hooksPath /dev/null || true
        git -C "${REPO_ROOT}" archive --format=tar HEAD \
            | tar -C "${STAGE_DIR}/LMCache" -xf -
        return 0
    fi
    copy_tree_without_vcs "${REPO_ROOT}" "${STAGE_DIR}/LMCache"
}

prepare_ascend_plugin_checkout

STAGE_DIR="$(mktemp -d)"
stage_lmcache_sources
copy_tree_without_vcs "${ASCEND_PLUGIN_ROOT}" "${STAGE_DIR}/LMCache-Ascend"

INNER="$(mktemp)"
cat > "${INNER}" <<'EOF'
set -euo pipefail
cp -a /src/LMCache /tmp/LMCache
cp -a /src/LMCache-Ascend /tmp/LMCache-Ascend
cd /tmp/LMCache

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND:-0.0.0+ci}"

export BUILD_WITH_ASCEND=1
export NO_GPU_EXT=1
export ENABLE_CXX11_ABI=1
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LMCACHE_TRACK_USAGE=false
export PYTHONHASHSEED=0
export PYTHONPATH="/tmp/pytest_site:${PYTHONPATH:-}"

if [[ -n "${ASCEND_CI_HTTP_PROXY:-}" ]]; then
  export http_proxy="${ASCEND_CI_HTTP_PROXY}"
  export https_proxy="${ASCEND_CI_HTTPS_PROXY:-${ASCEND_CI_HTTP_PROXY}}"
  export HTTP_PROXY="${http_proxy}"
  export HTTPS_PROXY="${https_proxy}"
  export no_proxy="${ASCEND_CI_NO_PROXY:-localhost,127.0.0.1}"
  export NO_PROXY="${no_proxy}"
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/devlib"

if [[ -d /usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc ]]; then
  export CPLUS_INCLUDE_PATH="/usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc:${CPLUS_INCLUDE_PATH:-}"
  export C_INCLUDE_PATH="/usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc:${C_INCLUDE_PATH:-}"
fi

python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r requirements/common.txt
python3 -m pip install -q -r requirements/test.txt
if [[ -f requirements/proto.txt ]]; then
  python3 -m pip install -q -r requirements/proto.txt
fi
unset NO_NATIVE_EXT NO_CUDA_EXT
rm -f lmcache/lmcache_native*.so 2>/dev/null || true
rm -rf build /tmp/LMCache-Ascend/build 2>/dev/null || true
python3 -m pip install -q --no-build-isolation -e /tmp/LMCache
python3 -m pip install -q --no-build-isolation -e /tmp/LMCache-Ascend

python3 -c '
import lmcache
import lmcache_ascend
print("lmcache ok", getattr(lmcache, "__version__", "unknown"))
print("lmcache_ascend ok", getattr(lmcache_ascend, "__version__", "unknown"))
'

python3 -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate

mkdir -p /tmp/pytest_site
cat > /tmp/pytest_site/sitecustomize.py <<'PYEOF'
import torch

try:
    import torch_npu  # noqa: F401 — required when TORCH_DEVICE_BACKEND_AUTOLOAD=0
except ImportError:
    pass

if getattr(torch, "npu", None) is not None and torch.npu.is_available():
    torch.npu.set_device(0)
    torch.zeros(1, device="npu")
PYEOF

set +e
python3 -u -m pytest -v \
  tests/v1/multiprocess \
  tests/inprocess/v1/storage_backend/test_p2p_backend_with_controller.py \
  tests/v1/distributed/l2_adapters/test_p2p_l2_adapter.py \
  -m "not cuda" \
  --ignore=tests/v1/multiprocess/test_client.py \
  --ignore=tests/v1/multiprocess/test_grpc_transport.py \
  -o log_cli=false \
  --junit-xml=/ci-out/ascend-junit.xml \
  --tb=short \
  -q
EXIT=$?
set -e
exit $EXIT
EOF
chmod 700 "${INNER}"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
[[ "${ASCEND_HOME_PATH}" =~ ^/[A-Za-z0-9._+/-]+$ ]] \
    || fail "ASCEND_HOME_PATH is invalid"
case "${ASCEND_HOME_PATH}" in
    *..*) fail "ASCEND_HOME_PATH is invalid" ;;
esac
case "${ASCEND_RT_VISIBLE_DEVICES:-0}" in
    *[[:space:]]*) fail "ASCEND_RT_VISIBLE_DEVICES is invalid" ;;
esac
case "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}" in
    *[[:space:]]*) fail "setuptools version override is invalid" ;;
esac
case "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND:-0.0.0+ci}" in
    *[[:space:]]*) fail "setuptools version override is invalid" ;;
esac
case "${ASCEND_CI_NO_PROXY:-localhost,127.0.0.1}" in
    *[[:space:]]*) fail "NO_PROXY value is invalid" ;;
esac

ENV_FILE="$(mktemp)"
chmod 600 "${ENV_FILE}"
{
    printf 'BUILD_WITH_ASCEND=1\n'
    printf 'NO_GPU_EXT=1\n'
    printf 'TORCH_DEVICE_BACKEND_AUTOLOAD=0\n'
    printf 'SOC_VERSION=%s\n' "${SOC_VERSION}"
    printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "${ASCEND_RT_VISIBLE_DEVICES:-0}"
    printf 'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=%s\n' \
        "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
    printf 'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND=%s\n' \
        "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND:-0.0.0+ci}"
    printf 'ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH}"
    if [[ -n "${ASCEND_CI_HTTP_PROXY:-}" ]]; then
        printf 'ASCEND_CI_HTTP_PROXY=%s\n' "${ASCEND_CI_HTTP_PROXY}"
        printf 'ASCEND_CI_HTTPS_PROXY=%s\n' "${ASCEND_CI_HTTPS_PROXY:-${ASCEND_CI_HTTP_PROXY}}"
        printf 'ASCEND_CI_NO_PROXY=%s\n' "${ASCEND_CI_NO_PROXY:-localhost,127.0.0.1}"
    fi
} > "${ENV_FILE}"

DOCKER_OPTS=(
    --rm
    --user root
    --security-opt no-new-privileges:true
    --label "owner=lmcache-buildkite-ascend"
    --env-file "${ENV_FILE}"
    -v "${STAGE_DIR}/LMCache:/src/LMCache:ro"
    -v "${STAGE_DIR}/LMCache-Ascend:/src/LMCache-Ascend:ro"
    -v "${INNER}:/tmp/run_pytest_inner.sh:ro"
    -v "${ARTIFACT_DIR}/junit:/ci-out"
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    -v /usr/local/dcmi:/usr/local/dcmi:ro
)

if [[ "${ASCEND_CI_IPC_HOST:-}" == "1" ]]; then
    log "Host IPC enabled by the agent opt-in"
    DOCKER_OPTS+=(--ipc=host)
else
    DOCKER_OPTS+=(--shm-size=8g)
fi

add_davinci_devices() {
    local list="${ASCEND_RT_VISIBLE_DEVICES:-0}"
    local device extra
    local -a parts
    IFS=',' read -ra parts <<< "${list}"
    [[ ${#parts[@]} -gt 0 ]] || fail "ASCEND_RT_VISIBLE_DEVICES is empty"
    for device in "${parts[@]}"; do
        [[ "${device}" =~ ^[0-9]+$ ]] || fail "ASCEND_RT_VISIBLE_DEVICES must be device indexes"
        [[ "${device}" -le 15 ]] || fail "ASCEND_RT_VISIBLE_DEVICES index is out of range"
        [[ -e "/dev/davinci${device}" ]] || fail "Davinci device node /dev/davinci${device} is missing"
        DOCKER_OPTS+=(--device "/dev/davinci${device}")
    done
    for extra in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
        [[ -e "${extra}" ]] || fail "required Ascend device node ${extra} is missing"
        DOCKER_OPTS+=(--device "${extra}")
    done
}
add_davinci_devices

if [[ -n "${ASCEND_CI_CA_CERT_FILE:-}" ]]; then
    case "${ASCEND_CI_CA_CERT_FILE}" in
        /*) ;;
        *) fail "CA certificate path must be absolute" ;;
    esac
    case "${ASCEND_CI_CA_CERT_FILE}" in
        *..*) fail "CA certificate path is invalid" ;;
    esac
    [[ -f "${ASCEND_CI_CA_CERT_FILE}" ]] || fail "CA certificate file is not readable"
    DOCKER_OPTS+=(
        -v "${ASCEND_CI_CA_CERT_FILE}:/etc/ssl/certs/ascend-ci-ca.pem:ro"
        -e "SSL_CERT_FILE=/etc/ssl/certs/ascend-ci-ca.pem"
        -e "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ascend-ci-ca.pem"
        -e "CURL_CA_BUNDLE=/etc/ssl/certs/ascend-ci-ca.pem"
    )
fi

if on_buildkite; then
    if [[ "${ASCEND_CI_ALLOW_OPEN_EGRESS:-}" == "1" ]]; then
        log "Open container egress is explicitly enabled on this agent"
    else
        case "${ASCEND_CI_DOCKER_NETWORK:-}" in
            ""|host|bridge|none)
                fail "set ASCEND_CI_DOCKER_NETWORK to a custom restricted Docker network, or ASCEND_CI_ALLOW_OPEN_EGRESS=1 on the agent"
                ;;
        esac
        [[ "${ASCEND_CI_DOCKER_NETWORK}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] \
            || fail "ASCEND_CI_DOCKER_NETWORK is invalid"
        docker network inspect "${ASCEND_CI_DOCKER_NETWORK}" >/dev/null \
            || fail "configured Docker network does not exist"
        DOCKER_OPTS+=(--network "${ASCEND_CI_DOCKER_NETWORK}")
        log "Using the agent restricted container network"
    fi
fi

log "Running Ascend NPU tests in ${IMAGE}"
docker run "${DOCKER_OPTS[@]}" "${IMAGE}" bash /tmp/run_pytest_inner.sh \
    || fail "Ascend NPU container failed"

log "Ascend NPU tests finished"
