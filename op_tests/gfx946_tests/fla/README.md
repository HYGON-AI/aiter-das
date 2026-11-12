# FLA 公开接口验证

使用 `test_fla.py --op NAME` 可单独验证各入口。`NAME` 支持
`sglang_chunk`、`vllm_chunk`、`chunk_o`、`packed_decode`、`sglang_gating`
和 `vllm_gating`。使用 `--list --suite full` 可列出完整用例，无需导入 GPU 运行库。

测试采用独立的 CPU 矩阵运算和递推公式作为参考，固定输入随机种子，逐元素断言数值，
并检查返回结构、存储别名及状态缓冲区中应保持不变的区域。测试不包含预热或计时循环。
失败时返回非零退出码，并保存状态为 `FAIL` 的 JSON 结果。

## 编译与运行

将 `AITER_REPO` 设置为已核对的容器内源码目录。使用性能模型时，必须在启动 Python
之前设置以下环境变量：

```bash
export ROCM_PATH=/opt/rocm PMD_PATH=/opt/rocm/pmd GPU_ARCHS=gfx946
export HIP_CLANG_PATH=/wksp/llvm-project/build-6.3.3/bin
export PATH=/opt/rocm/bin:$HIP_CLANG_PATH:/opt/hyhal/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/pmd:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$AITER_REPO" AITER_ENABLE_SUPERNODE_AR=0 MAX_JOBS=3
export GPU_ARGS="['--num-gpu=1']" GPU_DFLAGS="['StatLog']"
export AITER_FLA_TEST_SPLIT_LOG=1
unset AITER_JIT_DIR AITER_REBUILD HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
test_py="$AITER_REPO/op_tests/gfx946_tests/fla/test_fla.py"
GPU_CHIP=sb python "$test_py" --suite build --rebuild --json build.json
```

编译对象是完整的 `module_cpp_api`，包含其中的非 FLA 依赖。
首次干净构建后，`--refresh` 根据源码和头文件依赖重新编译受影响目标并重新链接；
`--rebuild` 则先删除该模块的构建目录再重建。使用前应确认没有其它任务正在构建同一模块。

以下每种运行配置都应使用独立工作目录，以保留各自的原始模型输出。运行结束后，
在对应目录执行调度检查；单 XCD 和普通启动对照不传 `--require-dual`：

```bash
GPU_CHIP=sb python "$test_py" --suite full --expect-xcds 1 --json result.json > run.log 2> launch.log
GPU_CHIP=sbx2 python "$test_py" --suite full --expect-xcds 2 --json result.json > run.log 2> launch.log
GPU_CHIP=sbx2 python "$test_py" --expect-xcds 2 --die-mode linear --chunk 2 --json result.json > run.log 2> launch.log
GPU_CHIP=sbx2 python "$test_py" --expect-xcds 2 --die-mode off --json result.json > run.log 2> launch.log
GPU_CHIP=sbx2 python "$test_py" --op packed_decode --expect-xcds 2 --json result.json > run.log 2> launch.log
python "$AITER_REPO/op_tests/gfx946_tests/fla/check_dispatch.py" "$PWD" --require-dual
```

在真实硬件上回归时，清除模型专用环境变量，使用已安装的工具链并选择空闲设备。
将 `GPU_ARCHS` 设置为实际架构，在各自对应的环境中执行：

```bash
python "$test_py" --suite full --require-arch gfx936 --rebuild --json hardware.json
python "$test_py" --suite full --require-arch gfx938 --rebuild --json hardware.json
```

## 接口约束与验收标准

- Chunk 输入要求 K=V=128，激活类型为 FP16/BF16，`chunk_size=64`。HIP 线程块包含
  256 个线程；函数名中的 `blockdim64` 后缀不代表实际启动的线程块大小。
- SGLang chunk 返回 `(h,v_new)`，即使 `output_final_state=False`，仍会更新有效状态槽。
  状态支持 FP32/BF16，槽间步长可以包含填充。负槽索引表示使用零初态，并跳过状态池写回。
  接口保留 `use_exp2` 参数，但原生实现使用自然指数。安全截断仅作用于 gate 差值；
  状态的 g/gk 衰减使用普通 `exp`。
- vLLM chunk 返回 `(h,v_new,final_state)`，后两项输出可分别控制是否返回。
  初始状态为 FP32，输入状态保持不变。
- Chunk output 计算分块内的因果注意力与分块初态贡献之和；覆盖标量 gate、
  每个 head 的 gamma、scale 和以 2 为底的指数运算。
- Packed decode 拆分 Q/K/V，返回与调用者 out/state 缓冲区共享存储的结果；
  对负槽索引将输出置零，并保持状态不变。
- SGLang gating 覆盖 GDN/KDA、归一化、禁止状态更新、树形缓存、带下界的 gate、
  beta 缩放及 ReplaySSM 数据缓存。
- vLLM gating 覆盖原地更新、packed 输入的非原地输出、KDA 和 accepted-token 初态选择。
  负输入槽索引对应的输出未定义，调用者不能将其视为已定义的零值。
  现有非原地状态输出按 T 分配，因此多序列测试使用 packed 输入。
- 调用者必须保证正索引不越界、可写状态槽不重复、packed 序列边界正确、树父节点合法。
  拒绝测试验证已有的元数据检查，不保证能拒绝任意不安全的索引内容。

JSON 中的 `PASS` 只表示数值和状态等测试断言通过。调度验收还需结合
`DUT_BEGIN`/`DUT_END`、FLA 启动日志和原始 dispatch/AQL 记录，定位
`m5out/0/<dispatch>/stats.txt`。按 XCD 汇总其所有 CU 的 `completedWGs`，
并断言总数等于完整三维启动 grid 的工作组数。每个入口至少要有一个代表用例，
证明两个 XCD 的完成计数都非零。普通启动模式 `off` 不等于单 XCD；
`--expect-xcds` 会查询并核对实际拓扑。模型结果不能证明真实硬件性能或 Graph 重放支持。

标准输出保存到 `run.log`，标准错误保存到 `launch.log`。启用日志分流设置后，
两个输出流都会记录相同的 DUT 标记。包装内部的输入转换 kernel 在最终 FLA kernel
启动前执行；解析器在每个经过同步的 DUT 区间内选取末尾对应的目标 dispatch，
先将完整 grid 和 block 与启动记录核对，再统计工作组数。
单 XCD 和普通启动对照应省略 `--require-dual`。
