import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    in_0 /= tmp_1
    tmp_3 = torch.nn.functional.dropout(in_0, 0.0, False, False)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def row_normalize_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load the entire row into registers
    x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)

    # Accumulate sum in float32 for numerical stability across dtypes
    x_f32 = x.to(tl.float32)
    row_sum = tl.sum(x_f32, axis=0)

    # Divide by the sum (L1 normalization), cast back to original dtype
    out = (x_f32 / row_sum).to(x.dtype)

    tl.store(output_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def fused_row_normalize_dropout(in_0):
    # Compute total number of rows and row length
    N = in_0.shape[-1]
    M = in_0.numel() // N

    out = torch.empty_like(in_0)

    row_normalize_kernel[(M,)](
        in_0,
        out,
        M,
        N,
    )

    return out


def replacement_func():
    return fused_row_normalize_dropout