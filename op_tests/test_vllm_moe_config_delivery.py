# SPDX-License-Identifier: MIT
"""Static checks for the vLLM-schema MoE configs owned by AITER.

These tests need neither a GPU nor an installed ``aiter`` package: the config
module is loaded from its source path, so they can run on a CPU-only host.

Run with ``python op_tests/test_vllm_moe_config_delivery.py`` or pytest.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_MODULE_PATH = (
    REPO_ROOT / "aiter" / "configs" / "vllm_moe" / "__init__.py"
)

TARGET_E = 44
TARGET_N = 1536
TARGET_K = 4096
TARGET_TOPK = 8
TARGET_DEVICE = "BW1100"
ALIAS_DEVICE = "BW1101"
TUNED_TILE = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 2,
    "num_stages": 1,
}
TUNED_MAX_M = 256


def _load_config_module():
    spec = importlib.util.spec_from_file_location(
        "aiter_vllm_moe_config", CONFIG_MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


moe_config = _load_config_module()


def _table(device_name: str = TARGET_DEVICE) -> dict[int, dict]:
    table = moe_config.load_config(
        TARGET_E, TARGET_N, device_name=device_name
    )
    assert table is not None
    return table


def test_target_shape_hits_for_both_device_spellings():
    for device_name in (TARGET_DEVICE, ALIAS_DEVICE):
        tile = moe_config.get_config(
            TARGET_E, TARGET_N, 128, device_name=device_name
        )
        assert tile == TUNED_TILE, device_name


def test_expected_target_parameters_are_returned():
    tile = moe_config.get_config(TARGET_E, TARGET_N, 128)
    assert tile is not None
    assert tile["BLOCK_SIZE_M"] == 16
    assert tile["BLOCK_SIZE_N"] == 64
    assert tile["BLOCK_SIZE_K"] == 128
    assert tile["GROUP_SIZE_M"] == 1
    assert tile["num_warps"] == 2
    assert tile["num_stages"] == 1


def test_nearest_m_bucket_matches_vllm_rule():
    table = _table()
    # 100 is nearer to 96 than to 128; 300 is nearer to 256 than to 512.
    assert moe_config.select_config(table, 100) == TUNED_TILE
    assert moe_config.select_config(table, 300) == TUNED_TILE
    # 65536 is vLLM's large-M default region.
    assert moe_config.select_config(table, 65536)["BLOCK_SIZE_M"] >= 64


def test_non_target_shapes_miss_and_keep_vllm_fallback():
    assert moe_config.get_config(352, TARGET_N, 128) is None
    # BOLTOPS' fused GEMM1 N (2 * 1536) is not vLLM's logical N.
    assert moe_config.get_config(TARGET_E, 3072, 128) is None
    assert moe_config.get_config(TARGET_E, TARGET_N + 1, 128) is None


def test_wrong_device_does_not_hit_the_target_config():
    for device_name in ("BW200B", "BW200", "BW9999", ""):
        assert (
            moe_config.get_config(
                TARGET_E, TARGET_N, 128, device_name=device_name
            )
            is None
        ), device_name


def test_wrong_dtype_family_does_not_hit_the_target_config():
    for dtype in ("fp8_w8a8", "int8_w8a8", "int4_w4a16"):
        assert (
            moe_config.get_config(
                TARGET_E, TARGET_N, 128, dtype=dtype
            )
            is None
        ), dtype


def test_block_shape_selector_does_not_hit_the_target_config():
    assert (
        moe_config.get_config(
            TARGET_E, TARGET_N, 128, block_shape=[128, 128]
        )
        is None
    )


def test_alias_files_are_byte_identical():
    primary = Path(
        moe_config.config_path(TARGET_E, TARGET_N, device_name=TARGET_DEVICE)
    )
    alias = Path(
        moe_config.config_path(TARGET_E, TARGET_N, device_name=ALIAS_DEVICE)
    )
    assert primary.is_file() and alias.is_file()
    assert primary.read_bytes() == alias.read_bytes()


def test_config_parser_loads_and_rejects_duplicate_buckets():
    table = _table()
    assert len(table) == 19
    assert sorted(table)[:3] == [1, 2, 4]
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as config_file:
        config_file.write('{"128": {"BLOCK_SIZE_M": 16}, "128": {}}')
        duplicate_path = config_file.name
    try:
        with open(duplicate_path, encoding="utf-8") as config_file:
            try:
                json.load(
                    config_file,
                    object_pairs_hook=moe_config._reject_duplicate_keys,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("duplicate M bucket was accepted")
    finally:
        os.unlink(duplicate_path)


def test_filename_contract_matches_vllm():
    assert (
        moe_config.config_file_name(TARGET_E, TARGET_N, device_name="BW1100")
        == "E=44,N=1536,device_name=BW1100.json"
    )
    assert (
        moe_config.config_file_name(
            TARGET_E, TARGET_N, device_name="BW 1100"
        )
        == "E=44,N=1536,device_name=BW_1100.json"
    )
    assert (
        moe_config.config_file_name(
            TARGET_E, TARGET_N, dtype="fp8_w8a8", device_name="BW1101"
        )
        == "E=44,N=1536,device_name=BW1101,dtype=fp8_w8a8.json"
    )


def test_native_boltops_key_cannot_resolve_this_table():
    """The BOLTOPS production Triton key is a different schema and stays a miss."""

    native_key = (
        f"E={TARGET_E},N={2 * TARGET_N},K={TARGET_K},topk={TARGET_TOPK},"
        f"arch=gfx938.json"
    )
    assert not (Path(moe_config.CONFIG_DIR) / native_key).exists()
    assert not (Path(moe_config.CONFIG_DIR) / native_key.replace(
        "arch=gfx938", "dtype=fp8_w8a8"
    )).exists()


def test_native_aiter_moe_config_directory_is_untouched():
    native = (
        REPO_ROOT
        / "aiter"
        / "ops"
        / "triton"
        / "configs"
        / "moe"
        / "E=128,N=352,device_name=BW200B.json"
    )
    assert native.is_file()


def test_prefill_rows_match_vllm_defaults_when_vllm_is_available():
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            get_default_config,
        )
    except Exception:
        return  # CPU host without vLLM; the static checks above still ran.

    table = _table()
    for bucket, tile in table.items():
        if bucket <= TUNED_MAX_M:
            assert tile == TUNED_TILE, bucket
            continue
        expected = get_default_config(
            bucket, TARGET_E, TARGET_N, TARGET_K, TARGET_TOPK, None
        )
        expected.pop("SPLIT_K")
        assert tile == expected, bucket


def _main() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001 - standalone harness
            failed += 1
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
        else:
            print(f"PASS {test.__name__}")
    print(f"STATIC_TESTS={len(tests)} FAILED={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
