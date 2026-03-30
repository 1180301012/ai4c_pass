import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: in_3 += in_0;  in_4 = in_3;  in_4 += in_2;  relu(in_4, inplace=True)
# Mirrors model.py exactly, minus the `= None` cleanup lines.
# ---------------------------------------------------------------------------
def pattern(in_0, in_2, in_3):
    in_3 += in_0
    in_4 = in_3
    in_4 += in_2
    tmp_0 = in_4
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    return tmp_2


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel: fused add+add+relu
# For input shapes [1,128,16,12] = 24 576 elements (fp16 / bf16).
# Fusing the two adds and relu saves 2 extra memory round-trips vs 3 separate
# kernels.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_add_add_relu_kernel(
    in0_ptr,
    in2_ptr,
    in3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x0 = tl.load(in0_ptr + offsets, mask=mask)
    x2 = tl.load(in2_ptr + offsets, mask=mask)
    x3 = tl.load(in3_ptr + offsets, mask=mask)

    result = x3 + x0 + x2
    zeros  = tl.zeros_like(result)
    result = tl.where(result > zeros, result, zeros)

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_add_relu(in_0, in_2, in_3):
    N = in_0.numel()
    out = torch.empty_like(in_0)

    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_add_add_relu_kernel[grid](
        in_0,
        in_2,
        in_3,
        out,
        n_elements=N,
    )
    return out


def replacement_func():
    return fused_add_add_relu