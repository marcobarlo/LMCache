#!/usr/bin/env bash
# Run Ascend NPU pytest in the vLLM-Ascend container on a self-hosted agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${BUILDKITE_BUILD_CHECKOUT_PATH:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
IMAGE="${ASCEND_CI_IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0rc1}"
SOC_VERSION="${SOC_VERSION:-Ascend910B4}"
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

if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>&1 | tee "${ARTIFACT_DIR}/host-npu-smi.txt" || true
fi

INNER="${ARTIFACT_DIR}/run_pytest_inner.sh"
cat > "${INNER}" <<EOF
set -euo pipefail
cd /workspace/LMCache

export SOC_VERSION="${SOC_VERSION}"
export NO_CUDA_EXT=1
export NO_GPU_EXT=1
export LMCACHE_TRACK_USAGE=false
export PYTHONHASHSEED=0
export PYTHONPATH="/tmp/pytest_site:\${PYTHONPATH:-}"

if getent hosts proxyhk.huawei.com >/dev/null 2>&1; then
  export http_proxy="http://${PROXY_HOST}"
  export https_proxy="http://${PROXY_HOST}"
fi

export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}:/usr/local/Ascend/ascend-toolkit/latest/\$(uname -i)-linux/devlib"

python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r requirements/test.txt
python3 -m pip install -q -r requirements/common.txt
python3 -m pip install -q -r requirements/proto.txt
python3 -m pip install -q -v --no-build-isolation -e .

python3 -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate

mkdir -p /tmp/pytest_site
cat > /tmp/pytest_site/sitecustomize.py <<'PYEOF'
import torch

if getattr(torch, "npu", None) is not None and torch.npu.is_available():
    torch.npu.set_device(0)
    torch.zeros(1, device="npu")
PYEOF

set +e
python3 -u -m pytest -v \\
  tests/v1/multiprocess \\
  tests/v1/storage_backend/test_p2p_backend_with_controller.py \\
  tests/v1/distributed/l2_adapters/test_p2p_l2_adapter.py \\
  -m "not cuda" \\
  --ignore=tests/v1/multiprocess/test_client.py \\
  --ignore=tests/v1/multiprocess/test_grpc_transport.py \\
  -o log_cli=false \\
  --timeout=180 \\
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
    --cap-add=SYS_PTRACE
    --label "owner=lmcache-buildkite-ascend"
    -v "${REPO_ROOT}:/workspace/LMCache:rw"
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
