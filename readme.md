## Installation
method build for develop:
```
git submodule update --init
GPU_ARCHS="gfx936" python -m pip install -e . --no-build-isolation --no-deps
```
method build for whl package:
```
bash rebuild_aiter.sh
```

If you happen to forget the `--recursive` during `clone`, you can use the following command after `cd aiter`
```
git submodule sync && git submodule update --init --recursive
```

## How to run the test cases
If you are in develop mode:
```
python op_tests/test_***.py
```
If you are in install mode:
```
python -m op_tests.test_***   # note that there is no '.py' suffix
```
## Aoubt the environment variable
1. 'AITER_LOG_MORE': log more info about aiter internal process,params, etc.
2. 'AITER_LOG_OP_PARAM': log the params for special interface, such as 'aiter_moe'.
3. `AITER_CODE_OBJECT_VERSION`: optional HIP code-object version (`4`, `5`, or `6`). It is also propagated to CK subprocesses and participates in the JIT cache key. The default is unset so existing ROCm environments keep their compiler default. The current yy-87 ROCm 7.2/gfx92a debug environment requires `AITER_CODE_OBJECT_VERSION=5`; this is an environment workaround, not a claim that the default/COV6 loader path is supported.


## Run operators supported by aiter

There are number of op test, you can run them with: `python3 op_tests/test_layernorm2d.py`
|  **Ops**                      | **Description**                                                                                                                                                   |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|ELEMENT WISE                   | ops: + - * /                                                                                                                                                      |
|SIGMOID                        | (x) = 1 / (1 + e^-x)                                                                                                                                              |
|AllREDUCE                      | Reduce + Broadcast                                                                                                                                                |
|KVCACHE                        | W_K W_V                                                                                                                                                           |
|MHA                            | Multi-Head Attention                                                                                                                                              |
|MLA                            | Multi-head Latent Attention with [KV-Cache layout](https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout )                                        |
|PA                             | Paged Attention                                                                                                                                                   |
|FusedMoe                       | Mixture of Experts                                                                                                                                                |
|QUANT                          | BF16/FP16 -> FP8/INT4                                                                                                                                             |
|RMSNORM                        | root mean square                                                                                                                                                  |
|LAYERNORM                      | x = (x - u) / (σ2 + ϵ) e*0.5                                                                                                                                      |
|ROPE                           | Rotary Position Embedding                                                                                                                                         |
|GEMM                           | D=αAβB+C                                                                                                                                                          |
