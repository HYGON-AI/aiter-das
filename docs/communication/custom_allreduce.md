<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
-->

# CustomAllreduce 用法与性能

`CustomAllreduce`（CAR）对组内各 rank 的张量做逐元素 SUM，返回同形状、同 dtype 的新张量，输入保持不变。本文介绍未量化的普通 AllReduce；AG/RS 和融合 RMSNorm 等接口见[通信接口说明](communication_ops.md)，量化通信见 [Quick AllReduce](quick_all_reduce.md)。HCU 的 PyTorch 设备名仍为 `cuda`。

## 接口与输入条件

```python
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

ca = CustomAllreduce(group, device, max_size=128 << 20,
                     enable_register_for_capturing=True)
supported = ca.should_custom_ar(x)
y = ca.custom_all_reduce(x, use_new=True, open_fp8_quant=False)
```

| 参数/接口 | 含义 |
|---|---|
| `group` | 用于交换元数据的非 NCCL ProcessGroup，例如 Gloo；每个 rank 绑定不同 GPU |
| `device` | 当前进程的 GPU，可传 `torch.device("cuda", local_rank)` |
| `max_size` | staging 容量，单位 byte，默认 1 GiB；普通 AR 的单 rank 输入上限还受 `max_size // 2` 限制 |
| `enable_register_for_capturing` | 请求 Graph 输入注册，默认 `True`；gfx936/gfx938 当前仍走受保护的 copy-in 路径 |
| `should_custom_ar(x)` | 检查 CAR 是否启用以及大小、布局、拓扑等准入条件；调用方仍需保证设备和 dtype 合法 |
| `custom_all_reduce(x)` | 自动分配输出；不满足准入时返回 `None`，调用方负责回退 |
| `all_reduce(x, out=out)` | 可复用预分配输出；绕过上层准入判断，使用前自行检查输入，输出须同设备、同形状、同 dtype，且不与输入重叠 |
| `capture()` | 包裹 `torch.cuda.graph(...)`，管理 CAR 的捕获状态和缓冲区注册 |
| `close()` | 释放 CAR 资源；调用前等待所有相关 GPU 工作完成并释放引用它的 Graph |

普通 AR 使用非空 FP16、BF16 或 FP32 张量。推荐 contiguous 布局，总字节数必须为 16 的倍数。所有 rank 必须按相同顺序调用，输入形状、dtype 和准入/回退决策一致；输出是 SUM，不会除以 world size。

`AITER_AR_MAX_SIZE_MB` 控制普通 AR 的大小阈值，默认 256 MiB，须为正整数。实际可接受大小为 `min(max_size // 2, 阈值)`，增大阈值不会自动扩容 staging。`max_size` 也不是对象的总显存占用：初始化另有约 `2 * max_size` 的 meta 区域及其他管理资源，应预留额外显存。

`AITER_AR_TRANSPORT=ipc` 是默认的单机路径；`fabric`/`auto` 需要相应编译能力和运行环境。普通 CAR 保留既有的 supernode 路径，本文的性能只覆盖单机 IPC、TP2。其他 rank 数和拓扑的能力不能从本表外推。

## 可直接运行的 eager 与 Graph 示例

保存为 `example_car.py`，使用两张卡运行：

```bash
HIP_VISIBLE_DEVICES=6,7 AITER_AR_TRANSPORT=ipc \
torchrun --standalone --nproc_per_node=2 example_car.py
```

```python
import os
import torch
import torch.distributed as dist
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", local_rank)
ca = CustomAllreduce(dist.group.WORLD, device, max_size=128 << 20,
                     enable_register_for_capturing=True)
graph = None
try:
    x = torch.full((4, 8192), rank + 1, dtype=torch.bfloat16, device=device)
    assert not ca.disabled and ca.should_custom_ar(x)
    expected = world * (world + 1) / 2

    y = ca.custom_all_reduce(x)
    torch.testing.assert_close(y, torch.full_like(x, expected), rtol=0, atol=0)

    # 真正的数值预热放在 ca.capture() 外，并使用 side stream。
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            y = ca.custom_all_reduce(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(graph):
            y = ca.custom_all_reduce(x)
    # 地址、shape、dtype 固定，内容可以在 replay 前更新。
    x.fill_(rank + 2)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y, torch.full_like(x, expected + world),
                               rtol=0, atol=0)
finally:
    torch.cuda.synchronize()
    graph = None
    dist.barrier()
    ca.close()
    dist.destroy_process_group()
```

### 回退与预分配输出

框架调用方可在 CAR 不支持时使用 RCCL。先在所有 rank 创建 `rccl_group = dist.new_group(backend="nccl")`，然后执行：

```python
y = ca.custom_all_reduce(x)
if y is None:
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM, group=rccl_group)
```

需要复用输出时，先验证 `ca.should_custom_ar(x)`，再使用 `out = torch.empty_like(x)` 和 `ca.all_reduce(x, out=out)`。同一对象的 staging 会复用，调用须按序执行；不要在多个流中并发操作同一个对象。

### Graph 注意事项

- 在 gfx936/gfx938 上，即使请求 `enable_register_for_capturing=True`，当前实现也使用 Graph 内 copy-in，以支持相同输入地址更新内容后的 replay；优化 one-stage kernel 的结束同步保证下一次 staging 写入前，各 rank 已完成读取。
- `AITER_AR_ENABLE_REG_CAPTURE` 是框架可能使用的开关；直接构造对象应传入上述参数。不要通过低层 native 接口强行绕过 copy-in 保护。
- `ca.capture()` 内、`torch.cuda.graph(...)` 外的调用仅返回用于模拟分配的零张量，不能把它当作实际规约结果。实际数值预热请放在 `ca.capture()` 外。
- Graph replay 需要各 rank 一致参与；保持输入地址和输出存活，结束使用后先同步 GPU，再释放 Graph 和 CAR，最后销毁进程组。

## 典型性能

测量日期：**2026-10-10**；算子源码版本：`9ec274d6`（包含 gfx936/gfx938 的 copy-in 和 one-stage 结束同步保护）。

| 项目 | gfx936 | gfx938 |
|---|---|---|
| 服务器 / 容器 | bw25 / zxw_sglang_1009 | nmz-2 / zxw_vllm_0928 |
| GPU / CU | BW200, UBB BW1000 / 80 CU | BW1100 / 72 CU |
| 软件 | DTK 26.04、HIP 6.3.26113、PyTorch 2.11.0 | DTK 26.04、HIP 6.3.26113、PyTorch 2.11.0 |
| 并行与模式 | 物理卡 6、7，TP2，IPC，未量化 | 物理卡 6、7，TP2，IPC，未量化 |

两端均使用 `use_new=True`、`open_fp8_quant=False`、`max_size=128 MiB`。gfx936 加载已验证的系统模块，gfx938 从相同提交独立编译；均保留 Fabric 编译能力，但测量使用 IPC。不同服务器、运行库和 GPU 配置的结果是各环境参考值，不用于隔离比较架构本身的快慢。

每组保留全部 30 个样本，逐样本取两 rank 较慢值再统计。Graph 每次 replay 连续执行 32 次操作，GPU event 时间除以 32；普通调用每样本执行一次并同步完成，报告墙钟耗时。初始化、编译和样本间 Gloo barrier 不计时。输入是精确可表示的八元素周期值，加 rank 和小幅偏移；计时前后检查 SUM、输入不变，以及原地址更新内容后的结果。

CAR 和 RCCL 都返回新输出；RCCL 对照为 `x.clone()` 后 `dist.all_reduce()`，**包含 clone 的 GPU 拷贝成本**，不代表原地 RCCL 的延迟。Graph 捕获阶段的 Python 分配不计时，replay 复用捕获的输出存储；eager 包含 Python 分派、输出分配、提交及完成同步。以下是算子微基准，不是模型吞吐或请求尾延迟。

### Graph 稳态延迟

单位 μs，表中为中位数；payload 为**每 rank 输入字节数**。最后一列是同形状 BF16 的 RCCL 对照。

| 架构 | payload | CAR FP16 | CAR BF16 | CAR FP32 | RCCL BF16（含 clone） |
|---|---|---:|---:|---:|---:|
| gfx936 | 4 KiB | 13.715 | 14.135 | 13.600 | 15.917 |
| gfx936 | 16 KiB | 14.055 | 14.225 | 13.660 | 16.650 |
| gfx936 | 24 KiB | 14.282 | 14.495 | 13.870 | 16.792 |
| gfx936 | 192 KiB | 17.612 | 18.645 | 17.532 | 20.270 |
| gfx936 | 512 KiB | 24.660 | 25.245 | 24.215 | 28.827 |
| gfx936 | 2 MiB | 42.747 | 43.280 | 42.947 | 54.370 |
| gfx936 | 8 MiB | 113.450 | 114.705 | 113.535 | 146.525 |
| gfx936 | 32 MiB | 386.897 | 389.237 | 386.525 | 493.177 |
| gfx938 | 4 KiB | 16.650 | 16.635 | 16.335 | 18.140 |
| gfx938 | 16 KiB | 16.837 | 16.807 | 16.195 | 19.207 |
| gfx938 | 24 KiB | 16.665 | 16.902 | 16.600 | 19.207 |
| gfx938 | 192 KiB | 18.332 | 18.812 | 18.705 | 24.002 |
| gfx938 | 512 KiB | 26.812 | 27.095 | 26.685 | 32.362 |
| gfx938 | 2 MiB | 43.495 | 44.300 | 43.607 | 49.822 |
| gfx938 | 8 MiB | 112.697 | 111.757 | 114.435 | 108.110 |
| gfx938 | 32 MiB | 388.572 | 379.892 | 395.900 | 350.620 |

BF16 的 P95、最大值和波动如下，仍以 Graph 内每次操作时间为单位。P95 是 30 个批次均值的 P95，不是单次请求 P95。

| 架构 | payload | CAR P95 μs | CAR max μs | CAR CV | RCCL CV |
|---|---|---:|---:|---:|---:|
| gfx936 | 4 KiB | 14.760 | 14.845 | 1.95% | 1.94% |
| gfx936 | 16 KiB | 14.865 | 15.800 | 2.37% | 2.23% |
| gfx936 | 24 KiB | 14.940 | 15.195 | 1.32% | 1.11% |
| gfx936 | 192 KiB | 20.110 | 21.130 | 3.59% | 4.16% |
| gfx936 | 512 KiB | 25.800 | 26.460 | 1.19% | 0.95% |
| gfx936 | 2 MiB | 44.140 | 46.985 | 1.61% | 4.74% |
| gfx936 | 8 MiB | 119.290 | 119.410 | 1.11% | 1.57% |
| gfx936 | 32 MiB | 391.190 | 404.710 | 0.72% | 0.15% |
| gfx938 | 4 KiB | 17.590 | 17.630 | 1.85% | 1.51% |
| gfx938 | 16 KiB | 17.600 | 17.855 | 2.17% | 1.78% |
| gfx938 | 24 KiB | 17.665 | 17.710 | 1.71% | 1.54% |
| gfx938 | 192 KiB | 19.285 | 19.420 | 1.03% | 0.87% |
| gfx938 | 512 KiB | 28.040 | 28.615 | 1.54% | 6.20% |
| gfx938 | 2 MiB | 45.165 | 45.245 | 0.87% | 8.77% |
| gfx938 | 8 MiB | 112.805 | 112.835 | 0.39% | 0.37% |
| gfx938 | 32 MiB | 381.950 | 387.130 | 0.38% | 0.10% |

### 普通调用延迟（BF16）

单位 μs，包含主机调用和等待完成；不能与上面的 Graph GPU 时间直接相减推导 kernel 开销。

| 架构 | payload | CAR median | CAR P95 | CAR CV | RCCL median（含 clone） |
|---|---|---:|---:|---:|---:|---:|
| gfx936 | 4 KiB | 159.725 | 171.979 | 3.59% | 165.290 |
| gfx936 | 2 MiB | 169.035 | 202.339 | 6.03% | 177.644 |
| gfx936 | 32 MiB | 528.943 | 571.168 | 3.12% | 581.948 |
| gfx938 | 4 KiB | 161.435 | 176.440 | 14.67% | 168.745 |
| gfx938 | 2 MiB | 172.800 | 263.370 | 44.46% | 179.365 |
| gfx938 | 32 MiB | 530.314 | 559.410 | 2.01% | 447.264 |

### 结果解读与适用边界

- 在这组 BF16 Graph 数据中，gfx936 的 CAR 中位延迟低于包含 clone 的 RCCL 对照。gfx938 在 4 KiB～2 MiB 时 CAR 较低；8 MiB 时 CAR 为 111.757 μs、RCCL 为 108.110 μs，32 MiB 时分别为 379.892 / 350.620 μs，CAR 约慢 3.4% / 8.3%。容量准入只表示可执行，不保证在该容量下更快，应按实际环境和负载决定框架路由。
- gfx936 FP16 24 KiB 的 CAR Graph CV 为 17.27%，存在长尾，不能据其中位数宣称稳定收益。部分 RCCL 小载荷和 eager 组也有明显波动；原始样本全部保留，未删除异常点或选择性重测。
- 每架构 96 组（3 dtype × 8 payload × 2 后端 × 2 执行方式）均通过计时前后及输入更新检查。gfx938 另通过 15,360 次连续输入复用输出检查；两架构均实际运行了本文示例。这些合成输入检查不替代完整模型精度评估。
- 本表仅覆盖 TP2、单机 IPC、未量化 SUM，不能外推 TP4/TP8、跨节点 Fabric、量化路径或融合算子，也没有测量旧版不安全路径的加速比。

可复用的性能入口为 `op_tests/multigpu_tests/test_communication_api_perf.py --suite car`，启动方法见[测试说明](../../op_tests/multigpu_tests/README.md)。
