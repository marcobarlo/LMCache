# Buildkite Web UI Setup: Ascend 910B NPU Unit Tests

**Steps editor**: paste the contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "npu" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

This pipeline runs multiprocess and P2P pytest on Ascend 910B hardware while
making PR admission opt-in via the `npu` label (same role as `amd` / `musa`).

### Trigger strategy

| Condition | Result |
|-----------|--------|
| PR label includes `npu` | upload the Ascend pipeline |
| PR label includes `full` | upload the Ascend pipeline |
| branch is `dev` | upload the Ascend pipeline |
| any docs/asset-only change | path filter skips upload |
| any change under `.buildkite/k3_tests/ascend/` | path filter forces upload |

The path filter treats Markdown, licenses, repository metadata, and files under
`docs/`, `asset/`, and `.github/` as trivial. Add the `force-ci` label when a
docs-only PR still needs the Ascend lane.

## Pipeline settings

| Pipeline slug (suggested) | Steps editor source | Uploaded definition |
|---------------------------|---------------------|-------------------|
| `lmcache-ascend-910b` | `buildkite-pipeline.yml` | `.buildkite/k3_tests/ascend/pipeline.yml` |

Both the upload step and the uploaded test step target queue **`Ascend910B`**.
The repository does not create that queue; register it in the Buildkite UI and
assign your self-hosted agent to it.

Concurrency group **`lmcache/ascend-910b`** ensures only one NPU job uses the
device at a time.

Optional overrides:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ASCEND_CI_IMAGE` | `quay.io/ascend/vllm-ascend:v0.26.0rc1` | Container image |
| `SOC_VERSION` | `Ascend910B4` | Passed into the container |
| `NO_CUDA_EXT` | `1` | Set on `docker run` and before `pip install -e .` (no CUDA extension build) |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | `0` | Required on vLLM-Ascend / PyTorch 2.x so CUDA backends are not auto-loaded on NPU hosts |
| `ASCEND_CI_ARTIFACT_DIR` | `ascend-ci-artifacts` | Host directory for logs and JUnit |
| `ASCEND_CI_PROXY_HOST` | `host.docker.internal:3128` | HTTP proxy when `proxyhk.huawei.com` resolves |

## Required agent setup

The self-hosted agent must provide:

1. A Linux Ascend 910B host with working driver and toolkit.
2. Docker access and permission to use Davinci devices from containers.
3. Devices `/dev/davinci0`, `/dev/davinci1`, `/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc`.
4. Read-only mounts of `/usr/local/Ascend/driver`, `/usr/local/dcmi`, and `npu-smi`.
5. The `buildkite-agent` binary registered to queue **`Ascend910B`** (outbound
   connection to Buildkite; same model as MUSA `MooreThreads` and AMD bare metal).

Install the agent on the intranet host; do not register this machine as a
GitHub Actions runner for public `LMCache/LMCache` pull requests.

## Buildkite UI snippet

```yaml
agents:
  queue: "Ascend910B"

steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/ascend/pipeline.yml
```

## What this pipeline runs

Inside the container:

- `pip install -e .` with `NO_CUDA_EXT=1` and `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
- gRPC stub generation for multiprocess transport tests
- NPU context init before pytest (avoids `aclrtMallocHost` 107002 on pinned alloc)
- `pytest` on `tests/v1/multiprocess`, P2P backend and adapter tests, `-m "not cuda"`

Artifacts: `ascend-junit.xml`, files under `ascend-ci-artifacts/`.
