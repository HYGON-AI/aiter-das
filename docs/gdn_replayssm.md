<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# Compact GDN speculative verification

`aiter.ops.triton.fla.gdn_replayssm.gdn_replayssm_verify` verifies a linear
draft chain while retaining a compact ReplaySSM ring. It does not allocate
or write a full recurrent-state snapshot for every draft token.

The measured target is gfx936, BF16 activations, FP32 checkpoints, K=V=128,
H=16, HV=48, a 16-entry ring and four draft positions. Other architectures,
model shapes and full dataset accuracy require separate qualification.

## Algorithm and precision

The checkpoint has layout `[state_slot, HV, V, K]`, with K contiguous. The
committed ring contains D, normalized K and FP32 log-decay G. The verifier
reconstructs one state tile, then applies the short recurrent draft chain
in registers. It writes outputs and the speculative compact records.

With `history_dot=True`, two activation-dtype matrix products preserve the
FP32 decay scaling used to reconstruct the committed history. A high part
and a low part represent each scaled update. The draft recurrence and state
accumulation stay FP32. No checkpoint dtype change is involved.

The output reconstruction contract consumes the existing D/K high parts.
Low parts are retained for checkpoint commit. They must not independently
be interpreted as an error metric: a BF16 rounding boundary can change the
sign of a low part while the represented high+low value remains accurate.

## Ownership and integration contract

- Q/K have shape `[tokens, H, 128]`, V `[tokens, HV, 128]`, gates A/B
  `[tokens, HV]`. Inner head/width axes must be dense. Token strides may be
  padded, as may the checkpoint and cache slot strides.
- `query_start_loc` packs the per-request windows. State indices address
  checkpoints; replay indices independently address request-owned rings.
- The caller owns cursor advancement, accepted-prefix commit, capacity
  flush and tracking snapshots. It must ensure valid indices, nonoverlapping
  live ring rows and `write_pos + window_length <= max_cache_len`.
- The verifier never writes the checkpoint or cursors. An invalid state
  index (`<= null_block_id`) yields zero output and no ring writes. Empty
  windows perform no work.
- Invoke `is_flush_kernel=False` for non-flush rows. Before invoking
  `is_flush_kernel=True`, the caller must have folded their history into
  the checkpoint. The two invocations are routed on the GPU and can be
  captured in a Graph. No tensor values are copied to the host.
- Supplying raw V/K buffers plus a beta buffer selects exact raw staging.
  Supplying V/K low buffers without beta selects compact residual staging.
- This is a GDN linear-chain verifier. It does not implement KDA or tree
  speculative verification.

This entry is independent of SGLang. In SGLang 0.5.22 it replaces only the
verify/flush-output launch at the `_launch_gdn_spec` integration boundary;
the existing checkpoint commit and cursor management must remain intact.
The existing prefill/packed-decode AITER selector alone does not replace
ReplaySSM verification. Integrations should select this API explicitly,
after assigning each worker's device, and retain their existing fallback
for unqualified shapes.

## Reproduction

```bash
HIP_VISIBLE_DEVICES=0 python -m pytest -q \
  op_tests/triton_tests/test_gdn_replayssm.py

HIP_VISIBLE_DEVICES=0 python \
  op_tests/op_benchmarks/triton/bench_gdn_replayssm.py \
  --batch 8 --history 8 --sglang-reference \
  --out results/gdn-b8-h8.json --dump-ir results/gdn-b8-h8-ir
```

`--sglang-reference` requires the matching SGLang 0.5.22 private launch
contract. Omit it for standalone AITER timing. Graph timing excludes cursor
advancement, checkpoint commit, full-state reset copies and model work;
it does not establish serving TTFT, TPOT or SLA by itself.
