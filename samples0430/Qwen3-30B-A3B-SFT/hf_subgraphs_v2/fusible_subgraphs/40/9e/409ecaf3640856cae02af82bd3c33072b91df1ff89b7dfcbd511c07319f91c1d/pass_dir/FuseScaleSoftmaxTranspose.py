import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=16),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def scale_softmax_transpose_kernel(
    input_ptr,
    output_ptr,
    scale,
    B,
    H,
    N,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """
    Fused: output[b, h, j, i] = softmax(scale * input[b, h, i, :])[-1][j]
    Each program handles one input row (b, h, i, :).
    Grid size = B * H * N rows.
    """
    row_id = tl.program_id(0)
    # Decompose row_id -> (b, h, i)
    i = row_id % N
    h = (row_id // N) % H
    b = row_id // (N * H)

    # Load input row: input[b, h, i, :]
    base_in = b * H * N + h * N + i * N
    j_offsets = tl.arange(0, BLOCK_N)
    mask = j_offsets < N

    x = tl.load(input_ptr + base_in + j_offsets, mask=mask, other=0.0).to(tl.float32)

    # Scale and compute softmax in fp32
    x = x * scale
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    x_sm = x_exp / x_sum

    # Write to output[b, h, j, i] = output_ptr + b*H*N + h*N + j*H + i
    base_out = b * H * N + h * N + i
    out_offsets = base_out + j_offsets * H
    tl.store(output_ptr + out_offsets, x_sm.to(OUT_DTYPE), mask=mask)


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return (tmp_2,)


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Replacement
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_scale_softmax_transpose(in_0):
    B = in_0.shape[0]
    H = in_0.shape[1]
    N = in_0.shape[2]

    # Output is the transposed softmax result: [B, H, N, N]
    output = torch.empty(B, H, N, N, dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (B * H * N,)

    if in_0.dtype == torch.float16:
        OUT_DTYPE = tl.float16
    elif in_0.dtype == torch.bfloat16:
        OUT_DTYPE = tl.bfloat16
    else:
        OUT_DTYPE = tl.float32

    scale_softmax_transpose_kernel[grid](
        in_0, output,
        0.1767766952966369,
        B, H, N,
        OUT_DTYPE=OUT_DTYPE,
    )

    return (output,)


def replacement_func():
    return fused_scale_softmax_transpose