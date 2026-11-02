#!/usr/bin/env bash
# Device-specific trigger condition and path rules for the Ascend NPU pipeline.
FILTER_DEVICE="ascend"
FILTER_TRIGGER_CONDITION='build.pull_request.labels includes "npu" || build.pull_request.labels includes "full" || build.branch == "dev"'

_path_filter_ascend_should_skip() {
    local changed_file="$1"

    case "$changed_file" in
        .buildkite/k3_tests/ascend/*)
            return 1
            ;;
        .buildkite/k3_tests/*/*)
            return 0
            ;;
        csrc/*|setup.py|setup_extensions/*|pyproject.toml|rust/*)
            return 1
            ;;
        lmcache/v1/multiprocess/*)
            return 1
            ;;
        lmcache/v1/distributed/l2_adapters/p2p_l2_adapter.py)
            return 1
            ;;
        lmcache/v1/storage_backend/p2p_backend.py)
            return 1
            ;;
        lmcache/v1/platform/devices/npu/*)
            return 1
            ;;
        tests/v1/multiprocess/*)
            return 1
            ;;
        tests/v1/storage_backend/test_p2p_backend_with_controller.py)
            return 1
            ;;
        tests/v1/distributed/l2_adapters/test_p2p_l2_adapter.py)
            return 1
            ;;
        tests/v1/platform/devices/npu/*)
            return 1
            ;;
        examples/*)
            return 0
            ;;
    esac

    return 0
}
