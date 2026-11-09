# Buildkite Web UI Setup: Ascend 910B NPU Unit Tests

Architecture and **outbound network** from the self-hosted agent: see [`CI_NETWORK.md`](CI_NETWORK.md).

**Steps editor**: paste the contents of `buildkite-pipeline.yml` **after** that file is on the protected base branch (`dev`, `main`, or `master`). The pasted step clones that branch and runs its bootstrap. It must not call `upload-pipeline.sh` or `run-in-container.sh` from the pull request checkout.

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
| `ASCEND_CI_IMAGE` | `quay.io/ascend/vllm-ascend:v0.26.0rc1` | Container image. Set on the agent, not in a pull request |
| `LMCACHE_ASCEND_CHECKOUT_PATH` | unset | Absolute host path of a pre-provisioned **LMCache-Ascend** checkout, outside the job checkout and outside `ascend-ci-artifacts`. Set in the **agent** environment. When unset, the runner caches a GitHub clone under the agent user's cache directory |
| `LMCACHE_ASCEND_REF` | `main` | Branch/tag/SHA used only for that cache clone |
| `LMCACHE_ASCEND_REPO` | `git@github.com:LMCache/LMCache-Ascend.git` | GitHub remote used only for that cache clone |
| `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE` | `0.0.0+ci` | Editable LMCache install skips git tag lookup (same as MUSA/AMD/harness CI) |
| `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND` | `0.0.0+ci` | Editable plugin install skips git tag lookup |
| `SOC_VERSION` | `Ascend910B4` | Passed into the container; use `Ascend910B3` on 910B3 hosts |
| `NO_GPU_EXT` | `1` | Skip GPU kernel extensions; common C++ (`lmcache_native`) still builds |
| `BUILD_WITH_ASCEND` | `1` | Select Ascend build profile (do **not** set `NO_CUDA_EXT=1`; that disables all native code) |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | `0` | Required on vLLM-Ascend / PyTorch 2.x so CUDA backends are not auto-loaded on NPU hosts |
| `ASCEND_CI_DOCKER_NETWORK` | required on Buildkite | Name of a **custom** Docker network whose egress the host firewall limits to the package index, image registry, and optional proxy. `host`, `bridge`, and `none` are rejected |
| `ASCEND_CI_ALLOW_OPEN_EGRESS` | unset | Set to `1` only in the agent environment when the host itself has no intranet path to protect. Do not set this in `pipeline.yml` |
| `ASCEND_CI_HTTP_PROXY` / `ASCEND_CI_HTTPS_PROXY` | unset | Optional `http://` or `https://` proxy for `pip`. Agent environment only; the script does not probe internal DNS names |
| `ASCEND_CI_CA_CERT_FILE` | unset | Optional single CA file mounted read-only when the proxy intercepts TLS |
| `ASCEND_CI_IPC_HOST` | unset | Set to `1` only if NPU tests cannot run with `--shm-size=8g`. Default is not host IPC |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | Comma-separated Davinci indexes to mount. Only those nodes are passed into Docker |

## Required agent setup

The self-hosted agent must provide:

1. A Linux Ascend 910B host with working driver and toolkit.
2. Docker access and permission to use Davinci devices from containers.
3. The Davinci node for each index in `ASCEND_RT_VISIBLE_DEVICES` (default `/dev/davinci0`), plus `/dev/davinci_manager`, `/dev/devmm_svm`, and `/dev/hisi_hdc`.
4. Read-only mounts of `/usr/local/Ascend/driver` and `/usr/local/dcmi`.
5. **`git`** on the agent when `LMCACHE_ASCEND_CHECKOUT_PATH` is unset.
6. A custom Docker network named by `ASCEND_CI_DOCKER_NETWORK`, with host firewall rules that limit that network's egress. The test container is not attached to the default bridge unless `ASCEND_CI_ALLOW_OPEN_EGRESS=1` is set on the agent.
7. The `buildkite-agent` binary registered to queue **`Ascend910B`**.

Install the agent on the host that owns the NPU. Do not register this machine as a
GitHub Actions runner for public pull requests. Keep `dev` (or `main`) protected:
host scripts run from that branch, so a pull request cannot change them until it is merged.

Changes to `run-in-container.sh`, `trusted-bootstrap.sh`, or `pipeline.yml` take effect on the agent only after they land on the base branch. Until the steps editor is updated to the bootstrap in `buildkite-pipeline.yml`, a pull request can still replace the host shell.

## Buildkite UI snippet

Paste `buildkite-pipeline.yml`. The command clones the base branch and runs `trusted-bootstrap.sh` from that clone. Do not replace it with a path inside the job checkout.

## What this pipeline runs

The container still runs as root because the vendor image and Davinci device nodes need it. It does not get `SYS_PTRACE`, host IPC (unless the agent opts in), or a bind mount of the job checkout:

- A copy of the tracked LMCache tree and the LMCache-Ascend tree is mounted **read-only** (no `.git`)
- The container copies those trees to its local disk, then `pip install -e` both with `BUILD_WITH_ASCEND=1`, `NO_GPU_EXT=1`, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
- Import check: `lmcache` + `lmcache_ascend`
- NPU context init before pytest (avoids `aclrtMallocHost` 107002 on pinned alloc)
- `pytest` on `tests/v1/multiprocess`, P2P backend and adapter tests, `-m "not cuda"`

Artifacts are an allowlist: `ascend-ci-artifacts/junit/ascend-junit.xml`, `ascend-ci-artifacts/host-npu-smi.txt`, and `ascend-ci-artifacts/failure.log`. `npu-smi` output is uploaded with IPv4, IPv6, and MAC addresses redacted. PCI bus ids in the health table are kept. The plugin source tree is not an artifact.
