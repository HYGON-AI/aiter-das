<!-- SPDX-License-Identifier: Apache-2.0 AND MIT -->
<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# SGLang GDN prefill state recursion

`aiter.ops.triton.fla.sglang.chunk_delta_h.chunk_gated_delta_rule_fwd_h`
accepts caller-supplied `chunk_indices` and `chunk_offsets`. Passing both
avoids rebuilding the metadata inside every state-recursion call and makes
the call capturable in a Graph. Metadata must describe the current packed
sequence lengths; the caller owns updates to it.

If chunk indices are omitted, the existing convenience path builds them by
reading sequence lengths on the host. Prepare metadata outside Graph capture
or pass metadata already constructed by the serving backend.

`initial_state` uses `[slots, value_heads, V, K]` with dense inner axes. Its
slot stride may be padded or span a larger envelope: addressing uses the
actual `stride(0)` and a 64-bit index product. A negative state index starts
from zero and does not write back. Without an index tensor, sequence n uses
state row n. `inplace_update=False` leaves the state pool unchanged while
still returning chunk states and updated values.

For the gfx936 K=V=128, BT=64, H=48 state-recursion shape, the configuration
table selects BV=32, four warps and one pipeline stage. Other shape entries
and architecture tables retain their existing settings. This is a fixed
configuration; it does not benchmark multiple configurations against a live
in-place state pool.

The return contract stays `(chunk_states, updated_values)`. Chunk states are
stored in the activation dtype and the recurrent carry keeps its existing
FP32 computation. This stage alone is not an entire GDN prefill pipeline;
normalization, intra-chunk transforms and output projection remain separate.

```bash
HIP_VISIBLE_DEVICES=0 python -m pytest -q \
  op_tests/triton_tests/test_gdn_prefill_metadata.py
```

Tests cover precomputed metadata, packed tails and empty sequences, padded
slot strides, null indices, missing/dense initial state, disabled write-back
and Graph replay against an independent recurrence.
