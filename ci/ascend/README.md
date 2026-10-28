# Ascend NPU CI (Jenkins)

Intranet Jenkins runs LMCache upstream tests on company Ascend hardware. GitHub
Actions on `LMCache/LMCache` does not use self-hosted runners for public pull
requests. The controller polls GitHub outbound; agents on 910B hosts connect
outbound to the controller.

## Controller (intranet VM)

1. Install Jenkins LTS (package or container) on a VM with outbound HTTPS to
   `api.github.com` and `github.com`. Do not expose the controller UI to the
   public internet (VPN or corp network only).
2. Install plugins: **Pipeline**, **Credentials**, **Lockable Resources**, **Timestamper** (optional).
3. Create a credential `github-lmcache-status` (username + fine-grained PAT or
   classic token) with **Contents: read** and **Commit statuses: write** on
   `LMCache/LMCache`.
4. Create two Pipeline jobs (both use the **Jenkinsfile on branch `dev` only**;
   never load the Jenkinsfile from a PR head):
   - **`lmcache-ascend-nightly`**: cron `H 2 * * *`, shell step runs
     [`run_nightly_trigger.sh`](run_nightly_trigger.sh) (or pass `MODE=nightly`
     and `GIT_SHA=origin/dev` manually).
   - **`lmcache-ascend-pr`**: cron every 10 minutes runs
     [`github_poll.py`](github_poll.py) with `--repo-root` pointing at a `dev`
     checkout on the controller.
5. Job settings: **Do not allow concurrent builds**, **Discard old builds**.
6. **Manage Jenkins → Lockable Resources:** create **`ascend-910b-npu`** (quantity 1).

Example poll cron (execute on controller, `dev` checkout at `/var/lib/jenkins/lmcache-dev`):

```bash
GITHUB_TOKEN=... JENKINS_USER=... JENKINS_TOKEN=... \
  python3 /var/lib/jenkins/lmcache-dev/ci/ascend/github_poll.py \
  --repo-root /var/lib/jenkins/lmcache-dev \
  --jenkins-url https://jenkins.example.corp
```

## Agent (910B Ascend host)

1. Label the node: `ascend`, `arm64`, `910b`.
2. Run the inbound agent as a service (see [`jenkins-agent.service.example`](jenkins-agent.service.example)).
3. Install Docker on the agent; the user running the agent must run containers
   with Davinci devices (same as LMCache-Ascend GHA).
4. **One NPU job at a time**: enable **# of executors = 1** on this node and
   use the `lock(resource: 'ascend-910b-npu')` step in the Jenkinsfile.

Do **not** register this host as a GitHub Actions runner for `LMCache/LMCache`.

## Pull request gate

Hardware runs for PRs only when either:

- Label **`npu`** is present, or
- An **APPROVED** review from someone with **write** access on the repo exists
  for the PR head SHA.

Nightly runs track **`dev`** only. Status context on GitHub: **`Ascend NPU`**
(informational; not a required check).

## Local dry run

From a checkout on an Ascend host with Docker:

```bash
export GIT_SHA="$(git rev-parse HEAD)"
export MODE=nightly
export PYTEST_PATHS="tests/v1/multiprocess tests/v1/storage_backend/test_p2p_backend_with_controller.py tests/v1/distributed/l2_adapters/test_p2p_l2_adapter.py"
./ci/ascend/run_in_npu_container.sh
```
