# AITER for Hygon GPUs

## Upstream attribution and licensing

This project is derived from [ROCm/aiter](https://github.com/ROCm/aiter), upstream branch `main`.
The fixed upstream reference for this compliance review is commit
[`552f4b85124b77b22db40223209ac1e19140d4d6`](https://github.com/ROCm/aiter/tree/552f4b85124b77b22db40223209ac1e19140d4d6).
Later upstream imports and Hygon changes are recorded in the Git history; this reference does not
claim that every file is identical to that commit.

The original upstream MIT license and copyright notice are preserved in [LICENSE](LICENSE).
Third-party portions retain their original licenses, including Apache-2.0 where indicated in the source.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream and component attribution, licenses, dependencies, and the separate list of source files whose provenance remains unresolved.

Modified by Hygon Information Technology Co., Ltd. for Hygon GPU support.
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
This contribution notice applies to Hygon's original additions and substantive modifications,
including GPU kernels, operator integration, communication, and JIT/build support.
Hygon's original contributions are provided under the MIT license, subject to the original
licenses of incorporated third-party code. Original upstream and third-party ownership is retained.

## Installation
Python 3.10 or newer is required. The build tools addressed by the compliance review
are pinned to `setuptools==79.0.1`, `setuptools_scm==10.2.1`, and `ninja==1.11.1`.
For the commands below, which disable build isolation or dependency installation,
install the pinned tools in the active Python environment first:

```bash
python -m pip install "setuptools==79.0.1" "setuptools_scm[toml]==10.2.1" "ninja==1.11.1"
```

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

## Continuous integration

GitHub Actions workflow definitions are in `.github/workflows/`; CI test source
files remain in `op_tests/ci_tests/`. The GPU build-and-test workflow runs only
on self-hosted runners in the `ci-general` Runner Group with the `ci` label.
The workflow runs inside the DTK/Python 3.10 container declared in
`.github/workflows/ci.yml`; the Runner passes through the HCU device nodes and
the read-only Hygon runtime mount. For the current SGLang CI image,
`optest` is the only package that needs an organisation-managed source today:
configure `OPTEST_PIP_INDEX_URL` (and, if needed, `OPTEST_PIP_TRUSTED_HOST`) as
a GitHub Actions Variable. The image lacks `optest`, `auditwheel`, and
`patchelf`; the latter two are explicitly fetched from `https://pypi.org/simple`.
It already provides
Torch 2.11, Triton, BoltOps, and the project runtime dependencies, which this
workflow deliberately does not reinstall. Once `optest` is preinstalled in the
SGLang image, its two Variables can be removed as well.
`AICC_NIGHTLY_DIR` is optional and only installs a newer AICC package when
configured. Set `DTK_PKG` only when running the environment script
outside that image and a replacement DTK archive is required.

For an interactive local reproduction on a BW1100 host, run
`bash ci_script/run_sglang_ci_container.sh`. Override `AITER_CI_IMAGE` or
`AITER_CI_CONTAINER_NAME` when necessary; attach with
`docker exec -it <container-name> /bin/bash`.

Contributions from forks are verified through pull requests to `main`. The
self-hosted CI Runner pool is network-isolated for this purpose.
The workflow checks out GitHub's immutable pull-request merge ref.
Test reports are published as GitHub Actions artifacts for each CI run.
CI therefore validates the result proposed for merge into `main`.

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
