import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: ATen non-inplace silu + detach (fallback for graphs where
# silu_ was converted to silu.default).
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.ops.aten.silu.default(in_0)
    tmp_1 = torch.ops.aten.detach.default(in_1)
    tmp_2 = torch.ops.aten.detach.default(in_2)
    tmp_3 = torch.ops.aten.detach.default(tmp_0)
    return (tmp_1, tmp_2, tmp_3, tmp_0)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel – element-wise SiLU: out = x * sigmoid(x)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},   num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048},  num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096},  num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192},  num_warps=16),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16),
    ],
    key=["n_elements"],
)
@triton.jit
def _silu_noninplace_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    result_f32 = x_f32 * tl.sigmoid(x_f32)
    result = result_f32.to(x.dtype)

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_silu_noninplace_forward(in_0, in_1, in_2):
    n = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _silu_noninplace_kernel[grid](in_0, out, n)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = out.detach()
    return (tmp_1, tmp_2, tmp_3, out)


def replacement_func():
    return triton_silu_noninplace_forward