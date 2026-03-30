import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pass 1: Match sum(dim=-1) → replace with a fast Triton row-sum kernel.
# Shape: [1,16,196,196] → [1,16,196]
# This replaces just the sum; Pass 2 (FuseRowNormalizeDropout) then fuses
# the unsqueeze+div using the sum result.
# ---------------------------------------------------------------------------

def pattern(in_0):
    return in_0.sum(dim=-1)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def row_sum_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    stride_in,
    stride_out,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N

    x = tl.load(in_ptr + row_idx * stride_in + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    row_sum = tl.sum(x_f32, axis=0)

    tl.store(out_ptr + row_idx * stride_out, row_sum.to(x.dtype))


@torch.fx.wrap
def triton_row_sum(in_0):
    shape = in_0.shape
    N = shape[-1]           # 196
    M = in_0.numel() // N  # 3136

    x = in_0.contiguous()
    out_shape = shape[:-1]  # [1, 16, 196]
    out = torch.empty(out_shape, device=in_0.device, dtype=in_0.dtype)

    row_sum_kernel[(M,)](
        x, out,
        M, N,
        N,  # stride_in (elements per input row)
        1,  # stride_out (1 output value per row)
    )

    return out


def replacement_func():
    return triton_row_sum