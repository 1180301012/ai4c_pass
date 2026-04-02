"""
Pattern B: conv2d(in_3, in_1, in_0, ...) -> stack([x], 0) -> sum(0) -> cat([x, in_2], 1)

Same optimization as Pattern A — only input argument ordering differs.
GEMM writes directly to output[:, :OC, :], copy writes to output[:, OC:, :].
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['N', 'IC', 'OC', 'HW'],
)
@triton.jit
def conv1x1_gemm_into_b(
    w_ptr, b_ptr, x_ptr, out_ptr,
    N, IC, OC, HW,
    TOTAL_C, OC_OFFSET,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_n  = tl.program_id(1)

    num_n_tiles = tl.cdiv(HW, BLOCK_N)
    pid_m  = pid_mn // num_n_tiles
    pid_ni = pid_mn %  num_n_tiles

    m_offs = pid_m  * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_ni * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_step in range(0, tl.cdiv(IC, BLOCK_K)):
        k_offs = k_step * BLOCK_K + tl.arange(0, BLOCK_K)

        w = tl.load(w_ptr + m_offs[:, None] * IC + k_offs[None, :],
                    mask=(m_offs[:, None] < OC) & (k_offs[None, :] < IC),
                    other=0.0)

        x = tl.load(x_ptr + pid_n * IC * HW + k_offs[:, None] * HW + n_offs[None, :],
                    mask=(k_offs[:, None] < IC) & (n_offs[None, :] < HW),
                    other=0.0)

        acc = tl.dot(w, x, acc)

    b = tl.load(b_ptr + m_offs, mask=m_offs < OC, other=0.0)
    acc += b[:, None]

    out_c = OC_OFFSET + m_offs
    tl.store(
        out_ptr + pid_n * TOTAL_C * HW + out_c[:, None] * HW + n_offs[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=(m_offs[:, None] < OC) & (n_offs[None, :] < HW),
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['N', 'C_SRC', 'HW'],
)
@triton.jit
def copy_into_b(
    src_ptr, dst_ptr,
    N, C_SRC, HW,
    DST_TOTAL_C, DST_OFFSET,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    n = pid_nc // C_SRC
    c = pid_nc % C_SRC

    hw_start = pid_hw * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)
    mask     = hw_offs < HW

    data = tl.load(src_ptr + (n * C_SRC + c) * HW + hw_offs, mask=mask, other=0.0)

    dst_c = DST_OFFSET + c
    tl.store(dst_ptr + (n * DST_TOTAL_C + dst_c) * HW + hw_offs, data, mask=mask)


@torch.fx.wrap
def fused_conv_stack_sum_cat_b(bias, weight, conv_input, cat_input):
    """
    Replaces: conv2d(in_3,...) → stack([x],0) → sum(0) → cat([x, in_2], 1)
    replacement_args swaps: conv_input=in_3, cat_input=in_2
    """
    N, IC, H, W = conv_input.shape
    OC      = weight.shape[0]
    OC2     = cat_input.shape[1]
    HW      = H * W
    TOTAL_C = OC + OC2

    output = torch.empty((N, TOTAL_C, H, W),
                         dtype=conv_input.dtype, device=conv_input.device)

    grid_gemm = lambda meta: (
        triton.cdiv(OC, meta['BLOCK_M']) * triton.cdiv(HW, meta['BLOCK_N']),
        N,
    )
    conv1x1_gemm_into_b[grid_gemm](
        weight, bias, conv_input, output,
        N, IC, OC, HW,
        TOTAL_C, 0,
    )

    grid_copy = lambda meta: (
        N * OC2,
        triton.cdiv(HW, meta['BLOCK_HW']),
    )
    copy_into_b[grid_copy](
        cat_input, output,
        N, OC2, HW,
        TOTAL_C, OC,
    )

    return output


def pattern(in_0, in_1, in_2, in_3):
    """Matches: conv2d(in_3, in_1, in_0, ...) → stack → sum → cat with in_2"""
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = torch.stack([conv2d], dim=0)
    tmp_4  = tmp_3.sum(dim=0)
    tmp_5  = torch.cat([tmp_4, in_2], 1)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_3, in_2)


def replacement_func():
    return fused_conv_stack_sum_cat_b