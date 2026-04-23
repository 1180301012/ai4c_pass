import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


def pattern(in_3):
    tmp_3 = in_3.mean(-2)
    return tmp_3


def replacement_args(in_3):
    return (in_3,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_F": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_F": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_F": 256}, num_warps=8, num_stages=2),
    ],
    key=["B"],
)
@triton.jit
def _mean49_kernel(
    x_ptr,
    out_ptr,
    B,
    x_s0,
    x_s1,
    x_s2,
    out_s0,
    out_s1,
    BLOCK_F: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_f = tl.program_id(1)

    offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    f_mask = offs_f < 448

    acc = tl.zeros([BLOCK_F], dtype=tl.float32)
    base = x_ptr + pid_b * x_s0 + offs_f * x_s2
    for t in tl.static_range(0, 49):
        vals = tl.load(base + t * x_s1, mask=f_mask, other=0.0)
        acc += vals.to(tl.float32)
    acc = acc * (1.0 / 49.0)
    tl.store(out_ptr + pid_b * out_s0 + offs_f * out_s1, acc, mask=f_mask)


@torch.fx.wrap
def mean_seq49_feature448(in_3):
    x = unwrap_tensor(in_3)
    return x.mean(-2)


def replacement_func():
    return mean_seq49_feature448