import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: match in_0[0] + index_select(-2, ...) → single tensor output
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    tmp_1 = in_0[0]
    tmp_2 = in_1.index_select(-2, tmp_1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel – BLOCK_M=2, BLOCK_D=16 → 32 elements = 1 full warp (100% utilisation)
# Grid: ceil(M/2) = 550 blocks, all resident simultaneously on A30 (28 SMs)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def gather_kernel(
    x_ptr,
    idx_ptr,
    out_ptr,
    M: tl.constexpr,          # always 1100 → compiler proves mask always True
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid    = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offs < M    # always True for valid pid < ceil(M/BLOCK_M)

    indices = tl.load(idx_ptr + m_offs, mask=mask_m, other=0)
    d_offs  = tl.arange(0, BLOCK_D)

    x_off  = indices[:, None] * BLOCK_D + d_offs[None, :]
    o_off  = m_offs[:, None]  * BLOCK_D + d_offs[None, :]
    mask2d = mask_m[:, None]

    rows = tl.load(x_ptr + x_off, mask=mask2d, other=0.0)
    tl.store(out_ptr + o_off, rows, mask=mask2d)


# ─────────────────────────────────────────────────────────────────────────────
# Host wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_index_select(in_0, in_1):
    M = in_0.shape[1]   # 1100
    D = in_1.shape[1]   # 16
    out = torch.empty(M, D, dtype=in_1.dtype, device=in_1.device)

    BLOCK_M = 2
    grid = ((M + BLOCK_M - 1) // BLOCK_M,)   # 550

    gather_kernel[grid](
        in_1, in_0, out,
        M=M, BLOCK_M=BLOCK_M, BLOCK_D=D,
        num_warps=1,
    )
    return out


def replacement_func():
    return triton_index_select