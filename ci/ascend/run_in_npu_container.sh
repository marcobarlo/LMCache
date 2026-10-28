#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run LMCache pytest inside the Ascend vLLM container (910B layout).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
GIT_SHA="${GIT_SHA:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
PYTEST_PATHS="${PYTEST_PATHS:-}"
JUNIT_OUT="${JUNIT_OUT:-${REPO_ROOT}/ascend-junit.xml}"

if [[ -z "${PYTEST_PATHS}" ]]; then
  PYTEST_PATHS="$(tr '\n' ' ' < "${REPO_ROOT}/ci/ascend/nightly_pytest_paths.txt" | xargs)"
fi

ASCEND_IMAGE="${ASCEND_IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0rc1}"
SOC_VERSION="${SOC_VERSION:-Ascend910B4}"
PROXY_HOST="${PROXY_HOST:-host.docker.internal:3128}"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

if ! git -C "$REPO_ROOT" cat-file -e "${GIT_SHA}^{commit}" 2>/dev/null; then
  git -C "$REPO_ROOT" fetch --depth=1 origin "${GIT_SHA}"
fi

git -C "$REPO_ROOT" archive "$GIT_SHA" | tar -x -C "$WORK_DIR"

SITE_DIR="${WORK_DIR}/.pytest_site"
mkdir -p "$SITE_DIR"
cat > "${SITE_DIR}/sitecustomize.py" <<'EOF'
import torch

if getattr(torch, "npu", None) is not None and torch.npu.is_available():
    torch.npu.set_device(0)
    torch.zeros(1, device="npu")
EOF

INNER="${WORK_DIR}/run_pytest_inner.sh"
cat > "$INNER" <<EOF
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
cp /tmp/sitecustomize.py /tmp/pytest_site/sitecustomize.py

echo ">>> pytest paths: ${PYTEST_PATHS}"
set +e
python3 -u -m pytest -v ${PYTEST_PATHS} \\
  -m "not cuda" \\
  --ignore=tests/v1/multiprocess/test_client.py \\
  --ignore=tests/v1/multiprocess/test_grpc_transport.py \\
  -o log_cli=false \\
  --timeout=180 \\
  --junit-xml=/workspace/LMCache/junit.xml \\
  --tb=short \\
  -q
EXIT=\$?
set -e
exit \$EXIT
EOF
chmod +x "$INNER"

DOCKER_OPTS=(
  --rm
  --user root
  --cap-add=SYS_PTRACE
  --label "owner=lmcache-jenkins-ascend"
  -v "${WORK_DIR}:/workspace/LMCache:rw"
  -v "${SITE_DIR}/sitecustomize.py:/tmp/sitecustomize.py:ro"
  -v "${INNER}:/tmp/run_pytest_inner.sh:ro"
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
  --device /dev/davinci0
  --device /dev/davinci1
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
)

if getent hosts proxyhk.huawei.com >/dev/null 2>&1; then
  DOCKER_OPTS+=(--add-host=host.docker.internal:192.168.1.1)
fi

docker run "${DOCKER_OPTS[@]}" "$ASCEND_IMAGE" bash /tmp/run_pytest_inner.sh
EXIT=$?

if [[ -f "${WORK_DIR}/junit.xml" ]]; then
  cp "${WORK_DIR}/junit.xml" "$JUNIT_OUT"
fi

exit $EXIT
