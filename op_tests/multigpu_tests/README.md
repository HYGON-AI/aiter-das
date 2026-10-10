# Communication tests

Run commands from the AITER repository root. Select the participating devices
with `HIP_VISIBLE_DEVICES` before launching a test. Each process must use a
different GPU. The direct API tests fail if the required backend is unavailable.

## Interface and correctness tests

| Test | Coverage | Processes |
|---|---|---|
| `test_custom_allreduce_api.py --suite collectives` | Dimension-aware AG/RS, integer bit preservation, input validation, changing Graph inputs, close/recreate | 2, 4, 8 |
| `test_custom_allreduce_api.py --suite fusions` | RMSNorm/Gemma, padding/strides, FP8 mirrors/scales, MXFP4 packing, QK norm, Graph registration | 2, 4, 8 |
| `test_custom_allreduce_api.py --suite transport` | IPC selection, collective initialization failures, topology rejection, optional disabled-Fabric check | 2, 4, 8; 6 with `--tp6` |
| `test_custom_allreduce_api.py --suite reuse` | Immediate input reuse, changing eager/Graph inputs, exact FP16/BF16/FP32 sums and HCU copy-in protection | 2, 4, 8 |
| `test_fused_ar_mhc_post.py` | AR+MHC one-stage/automatic/split entry points and Graph replay | 2, 4, 8 |
| `test_fused_qknorm_ar_rope.py` | Distributed QK statistics, RoPE, FP32 reference and fullgraph metadata | 2, 4, 8 |
| `test_parallel_groups.py` | TP/DP/PP/PCP/DCP/custom groups, public collectives, destruction and reinitialization | Exactly 4 |
| `test_quick_all_reduce_api.py --suite native` | Native QR handle APIs, quantization modes, BF16 cast policy, fused RMSNorm, Graph and resource reuse | 2, 4, 8 |
| `test_quick_all_reduce_api.py --suite communicator` | Python QR routing, admission checks, independent communicators and collective failure handling | 2, 4, 8 |
| `test_iris_comms.py --suite collectives` | Iris AG/RS, tail shapes, heap ownership/capacity and IPC lifetime | 2, 4, 8 |
| `test_iris_comms.py --suite fusion` | Iris RS/RMSNorm/FP8/AG fusion and independent FP8 conversion reference | 2, 4, 8 |
| `test_flydsl_quick_allreduce_int4.py` | Public INT4 API, small-payload BF16 policy, large payloads, Graph stress and performance | 2, 4, 8 |

For example:

```bash
HIP_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
  op_tests/multigpu_tests/test_custom_allreduce_api.py \
  --suite collectives --repeats 100 --close-repeats 20

HIP_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
  op_tests/multigpu_tests/test_custom_allreduce_api.py \
  --suite reuse --repeats 400

HIP_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  op_tests/multigpu_tests/test_parallel_groups.py

python op_tests/test_comm_fusion_meta.py
```

The metadata test checks shape/dtype/stride and `torch.compile(fullgraph=True)`
without launching GPU kernels. Iris tests need the optional dependency pinned
in `requirements-triton-comms.txt`; INT4 needs a compatible Hygon FlyDSL build.
Run each `--suite` in a separate torchrun invocation so its process-group and
environment settings are isolated. CAR collectives/fusions/reuse and native QR use
`--repeats` (default 100); Iris uses `--repeats` and collectives also uses
`--cycles` (default 20). CAR transport and QR communicator retain 100 Graph
replays. CAR fusions accepts `--register off/on/both`; native QR accepts `--mode`.
The IPC tests run on one node. Fabric transport tests require a compatible
build and device configuration; use `super_node/` for multi-node testing.

## Performance and broader regression

- `test_custom_allreduce_perf.py`: existing Custom AllReduce benchmark.
- `test_quick_all_reduce.py`: existing QR benchmark.
- `test_communication_api_perf.py`: representative TP2 CAR/fusion/MHC/QK-RoPE,
  HIP QR, Iris and parallel-group API latency. The `hip` suite measures GPU
  Graph time; `iris` and `groups_tp/groups_dp/groups_pp` measure synchronized
  eager wall and GPU time. Each case checks correctness before and after timing.
- `super_node/`: supernode benchmarks and transport/Graph tests.
- `../ci_tests/test_custom_allreduce.py`: FP16/BF16, shape and Graph/registration sweep.
- `../ci_tests/test_fused_ar_rms.py`: fused/split/quantized AR+RMSNorm correctness
  and timing. Use `-d fp16` or `-d bf16` to choose the dtype explicitly.
- The existing all-gather, reduce-scatter, QK norm, per-group quantization and
  GEMM/communication tests remain available for their respective APIs.

The INT4 test provides three suites:

```bash
HIP_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
  op_tests/multigpu_tests/test_flydsl_quick_allreduce_int4.py \
  --suite smoke --output /tmp/aiter_int4_smoke
```

Replace `smoke` with `large` for payloads through 320 MiB+16 B and 128 ordered
Graph replays, or `perf` for 32 KiB–64 MiB comparisons against HIP FP and RCCL.
`--grid-cap` limits the persistent grid. Results retain correctness metrics,
per-rank timing samples, collective median/P95/max and CV. With the default
small-payload policy, the public INT4 API uses HIP FP at <=32 KiB.

For the representative API benchmark, use two GPUs and provide an output directory:

```bash
AITER_ENABLE_SUPERNODE_AR=0 AITER_AR_TRANSPORT=ipc AITER_AR_ENABLE_REG_CAPTURE=0 \
AITER_QUICK_REDUCE_QUANTIZATION=NONE HIP_VISIBLE_DEVICES=6,7 \
torchrun --standalone --nproc_per_node=2 \
  op_tests/multigpu_tests/test_communication_api_perf.py \
  --suite hip --samples 30 --graph-ops 16 --output /tmp/comm_api_perf
```

Run each suite separately with a different output directory. Results include
all per-rank samples and collective median/P95/max/CV. Iris needs its optional
dependency; the group suites initialize AITER groups themselves. Usage and
published measurements are in [communication docs](../../docs/communication/communication_ops.md).
