from pathlib import Path

import pytest

from src.afd_environment import (
    RuntimeMetadata,
    environment_record,
    reject_cuda_env_interpreter,
    validate_runtime_pair,
)


COMMIT = "4c2541a53ec8ad507508073eac81ac9feecc6c9a"


def _runtime(*, old: bool) -> RuntimeMetadata:
    root = "/envs/cuda-env" if old else "/repo/third_party/afd-plugin/.venv"
    return RuntimeMetadata(
        python_version="3.10" if old else "3.12",
        python_executable=f"{root}/bin/python",
        python_base_prefix=root,
        python_include=f"{root}/include/python",
        python_h_available=True,
        vllm_version="0.15.1" if old else "0.26.0",
        vllm_root=f"{root}/lib/python/site-packages/vllm/__init__.py",
        torch_version="2.9.1" if old else "2.11.0",
        torch_root=f"{root}/lib/python/site-packages/torch/__init__.py",
        cuda_userspace_version="12.8" if old else "13.0",
        afd_plugin_version=None if old else "0.26.0rc2",
    )


def test_rejects_protected_cuda_env_as_afd_interpreter():
    with pytest.raises(ValueError, match="protected cuda-env"):
        reject_cuda_env_interpreter("/data/users/me/miniforge3/envs/cuda-env/bin/python")


def test_runtime_pair_requires_distinct_pinned_environments():
    validate_runtime_pair(
        _runtime(old=True),
        _runtime(old=False),
        expected_plugin_commit=COMMIT,
        plugin_commit=COMMIT,
    )


@pytest.mark.parametrize(
    ("old_version", "afd_version", "commit"),
    [("0.15.0", "0.26.0", COMMIT), ("0.15.1", "0.25.0", COMMIT),
     ("0.15.1", "0.26.0", "wrong")],
)
def test_runtime_pair_rejects_version_or_commit_drift(old_version, afd_version, commit):
    old = RuntimeMetadata(**{**_runtime(old=True).to_dict(), "vllm_version": old_version})
    afd = RuntimeMetadata(**{**_runtime(old=False).to_dict(), "vllm_version": afd_version})
    with pytest.raises(ValueError):
        validate_runtime_pair(
            old,
            afd,
            expected_plugin_commit=COMMIT,
            plugin_commit=commit,
        )


def test_runtime_pair_rejects_missing_afd_python_headers():
    afd = RuntimeMetadata(
        **{**_runtime(old=False).to_dict(), "python_h_available": False}
    )
    with pytest.raises(ValueError, match="development header"):
        validate_runtime_pair(
            _runtime(old=True),
            afd,
            expected_plugin_commit=COMMIT,
            plugin_commit=COMMIT,
        )


def test_environment_record_serializes_package_roots():
    record = environment_record(
        _runtime(old=True),
        _runtime(old=False),
        plugin_repository="https://example.invalid/afd.git",
        plugin_commit=COMMIT,
        plugin_clean=True,
        uv_version="uv 0.12.20",
    )
    assert "cuda-env" in record["protected_cuda_env"]["vllm_root"]
    assert "afd-plugin/.venv" in record["isolated_afd_env"]["vllm_root"]
    assert record["afd_plugin"]["clean"] is True
