#!/usr/bin/env bash
# Run Ascend NPU pytest in the vLLM-Ascend container on a self-hosted agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${BUILDKITE_BUILD_CHECKOUT_PATH:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
IMAGE="${ASCEND_CI_IMAGE:-quay.io/ascend/vllm-ascend:v0.26.0rc1}"
SOC_VERSION="${SOC_VERSION:-Ascend910B4}"
ASCEND_PLUGIN_REF="${LMCACHE_ASCEND_REF:-main}"
ASCEND_PLUGIN_REPO="${LMCACHE_ASCEND_REPO:-git@github.com:LMCache/LMCache-Ascend.git}"
ASCEND_PLUGIN_ROOT="${LMCACHE_ASCEND_CHECKOUT_PATH:-}"
PROXY_HOST="${ASCEND_CI_PROXY_HOST:-host.docker.internal:3128}"
ARTIFACT_PATH="${ASCEND_CI_ARTIFACT_DIR:-ascend-ci-artifacts}"
if [[ "${ARTIFACT_PATH}" == /* ]]; then
    ARTIFACT_DIR="${ARTIFACT_PATH}"
else
    ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_PATH}"
fi

log() {
    echo "--- :ascend: $*"
}

fail() {
    mkdir -p "${ARTIFACT_DIR}" 2>/dev/null || true
    printf '[ascend-ci] ERROR: %s\n' "$*" \
        | tee -a "${ARTIFACT_DIR}/failure.log" >&2
    exit 1
}

mkdir -p "${ARTIFACT_DIR}"
command -v docker >/dev/null 2>&1 || fail "docker is required on the Ascend agent"
command -v git >/dev/null 2>&1 || fail "git is required to prepare LMCache-Ascend"

if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>&1 | tee "${ARTIFACT_DIR}/host-npu-smi.txt" || true
fi

prepare_ascend_plugin_checkout() {
    if [[ -n "${ASCEND_PLUGIN_ROOT}" ]]; then
        [[ -d "${ASCEND_PLUGIN_ROOT}/.git" ]] \
            || fail "LMCACHE_ASCEND_CHECKOUT_PATH=${ASCEND_PLUGIN_ROOT} is not a git checkout"
        log "Using LMCache-Ascend checkout at ${ASCEND_PLUGIN_ROOT}"
        return 0
    fi
    ASCEND_PLUGIN_ROOT="${ARTIFACT_DIR}/LMCache-Ascend"
    if [[ ! -d "${ASCEND_PLUGIN_ROOT}/.git" ]]; then
        log "Cloning LMCache-Ascend (${ASCEND_PLUGIN_REF}) into ${ASCEND_PLUGIN_ROOT}"
        git clone --branch "${ASCEND_PLUGIN_REF}" --depth 1 "${ASCEND_PLUGIN_REPO}" \
            "${ASCEND_PLUGIN_ROOT}" \
            || fail "git clone LMCache-Ascend failed"
    else
        log "Updating LMCache-Ascend checkout at ${ASCEND_PLUGIN_ROOT}"
        git -C "${ASCEND_PLUGIN_ROOT}" fetch --depth 1 origin "${ASCEND_PLUGIN_REF}" \
            || fail "git fetch LMCache-Ascend failed"
        git -C "${ASCEND_PLUGIN_ROOT}" checkout -B ci-ascend-plugin "origin/${ASCEND_PLUGIN_REF}" \
            || fail "git checkout LMCache-Ascend ${ASCEND_PLUGIN_REF} failed"
    fi
    git -C "${ASCEND_PLUGIN_ROOT}" submodule update --init --recursive --depth 1 \
        || git -C "${ASCEND_PLUGIN_ROOT}" submodule update --init --recursive \
        || fail "LMCache-Ascend submodule init failed"
}

prepare_ascend_plugin_checkout

INNER="${ARTIFACT_DIR}/run_pytest_inner.sh"
cat > "${INNER}" <<EOF
set -euo pipefail
cd /workspace/LMCache

git config --global --add safe.directory /workspace/LMCache
git config --global --add safe.directory /workspace/LMCache-Ascend
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="\${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND="\${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND:-0.0.0+ci}"

export SOC_VERSION="${SOC_VERSION}"
export BUILD_WITH_ASCEND=1
export NO_GPU_EXT=1
export ENABLE_CXX11_ABI=1
export ASCEND_HOME_PATH="\${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LMCACHE_TRACK_USAGE=false
export PYTHONHASHSEED=0
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/tmp/pytest_site:\${PYTHONPATH:-}"

if getent hosts proxyhk.huawei.com >/dev/null 2>&1; then
  export http_proxy="http://${PROXY_HOST}"
  export https_proxy="http://${PROXY_HOST}"
fi

export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}:/usr/local/Ascend/ascend-toolkit/latest/\$(uname -i)-linux/devlib"

if [[ -d /usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc ]]; then
  export CPLUS_INCLUDE_PATH="/usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc:\${CPLUS_INCLUDE_PATH:-}"
  export C_INCLUDE_PATH="/usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc:\${C_INCLUDE_PATH:-}"
fi

python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r requirements/common.txt
python3 -m pip install -q -r requirements/test.txt
if [[ -f requirements/proto.txt ]]; then
  python3 -m pip install -q -r requirements/proto.txt
fi
unset NO_NATIVE_EXT NO_CUDA_EXT
rm -f lmcache/lmcache_native*.so 2>/dev/null || true
rm -rf build /workspace/LMCache-Ascend/build 2>/dev/null || true
python3 -m pip install -q -v --no-build-isolation -e /workspace/LMCache
python3 -m pip install -q -v --no-build-isolation -e /workspace/LMCache-Ascend

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
python3 -u -m pytest -v \\
  tests/v1/multiprocess \\
  tests/inprocess/v1/storage_backend/test_p2p_backend_with_controller.py \\
  tests/v1/distributed/l2_adapters/test_p2p_l2_adapter.py \\
  -m "not cuda" \\
  --ignore=tests/v1/multiprocess/test_client.py \\
  --ignore=tests/v1/multiprocess/test_grpc_transport.py \\
  -o log_cli=false \\
  --junit-xml=/workspace/LMCache/ascend-junit.xml \\
  --tb=short \\
  -q
EXIT=\$?
set -e
exit \$EXIT
EOF
chmod +x "${INNER}"

DOCKER_OPTS=(
    --rm
    --user root
    --ipc=host
    --cap-add=SYS_PTRACE
    --label "owner=lmcache-buildkite-ascend"
    -e "BUILD_WITH_ASCEND=1"
    -e "NO_GPU_EXT=1"
    -e "TORCH_DEVICE_BACKEND_AUTOLOAD=0"
    -e "SOC_VERSION=${SOC_VERSION}"
    -e "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}"
    -e "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
    -e "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND:-0.0.0+ci}"
    -e "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
    -v "${REPO_ROOT}:/workspace/LMCache:rw"
    -v "${ASCEND_PLUGIN_ROOT}:/workspace/LMCache-Ascend:rw"
    -v "${INNER}:/tmp/run_pytest_inner.sh:ro"
    -v "${ARTIFACT_DIR}:/ascend-ci-artifacts"
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    -v /usr/local/dcmi:/usr/local/dcmi:ro
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
    --device /dev/davinci0
    --device /dev/davinci1
    --device /dev/devmm_svm
    --device /dev/davinci_manager
    --device /dev/hisi_hdc
)

if [[ -d /etc/pki/ca-trust ]]; then
    DOCKER_OPTS+=(-v /etc/pki/ca-trust:/etc/pki/ca-trust:ro)
fi

if getent hosts proxyhk.huawei.com >/dev/null 2>&1; then
    DOCKER_OPTS+=(--add-host=host.docker.internal:192.168.1.1)
fi

log "Running Ascend NPU tests in ${IMAGE}"
docker run "${DOCKER_OPTS[@]}" "${IMAGE}" bash /tmp/run_pytest_inner.sh \
    || fail "Ascend NPU container failed"

if [[ -f "${REPO_ROOT}/ascend-junit.xml" ]]; then
    cp -f "${REPO_ROOT}/ascend-junit.xml" "${ARTIFACT_DIR}/" || true
fi

log "Ascend NPU tests finished"
