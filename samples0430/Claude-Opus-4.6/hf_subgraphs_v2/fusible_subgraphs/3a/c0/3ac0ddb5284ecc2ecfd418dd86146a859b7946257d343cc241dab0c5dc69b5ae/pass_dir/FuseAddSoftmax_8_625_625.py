import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 625, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 625, 625)
    tmp_4 = tmp_3.view(8, 625, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
    return (tmp_5, tmp_3)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_add_softmax_kernel_625(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    head_idx = row_idx // M
    m_idx = row_idx % M

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # in_0: [1, 1, M, N] contiguous - broadcast across heads
    in_0_offset = m_idx * N + col_offsets
    x0 = tl.load(in_0_ptr + in_0_offset, mask=mask, other=0.0).to(tl.float32)

    # in_1: [1, 8, M, N] contiguous
    in_1_offset = head_idx * (M * N) + m_idx * N + col_offsets
    x1 = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0).to(tl.float32)

    # Fused add
    x = x1 + x0

    # Softmax computation
    x = tl.where(mask, x, float('-inf'))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    x_softmax = x_exp / x_sum

    # Store output
    out_offset = row_idx * N + col_offsets
    tl.store(out_ptr + out_offset, x_softmax, mask=mask)


@torch.fx.wrap
def fused_add_softmax_8_625_625(in_0, in_1):
    M = 625
    N = 625
    num_heads = 8
    num_rows = num_heads * M

    # Allocate output as [1, 8, M, N]
    out_4d = torch.empty(1, num_heads, M, N, dtype=in_1.dtype, device=in_1.device)

    # Launch kernel - each program handles one row
    fused_add_softmax_kernel_625[(num_rows,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out_4d,
        M=M,
        N=N,
    )

    # Create views for both outputs
    out_3d = out_4d.view(num_heads, M, N)
    return (out_3d, out_4d)


def replacement_func():
    return fused_add_softmax_8_625_625