import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match the einsum contraction  'bchj,bhwj->bchw'
#
#   output[b,c,h,w] = sum_j( in_4[b,c,h,j] * in_1[b,h,w,j] )
#
# This is a batched matrix multiply where for each (b,h):
#   A = in_4[b, :, h, :]  →  [C, J]   (stride H*J in the c dimension)
#   B^T = in_1[b, h, :, :] →  [W, J]  (contiguous)
#   out[b, :, h, :] = A @ B^T →  [C, W]
#
# We replace the torch.functional.einsum call with a Triton tl.dot kernel
# that directly handles the non-standard memory layout of in_4 and avoids
# the einsum string-parsing overhead + cuBLAS dispatch latency.
# ---------------------------------------------------------------------------
def pattern(in_4, in_1):
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    return einsum


def replacement_args(in_4, in_1):
    return (in_4, in_1)


# ---------------------------------------------------------------------------
# Triton kernel
#
# Grid: (ceil(C / BLOCK_C), B * H)
#   pid_c  → which BLOCK_C-wide column of channels
#   pid_bh → which (b, h) pair
#
# Each CTA loads:
#   A tile:     in_4[b, c_range, h, :]  → [BLOCK_C, BLOCK_J]   (j innermost → coalesced)
#   B_natural:  in_1[b, h, :, :]        → [BLOCK_W, BLOCK_J]   (j innermost → coalesced)
# B_natural is then transposed in registers (tl.trans) to [BLOCK_J, BLOCK_W]
# for tl.dot, giving fully coalesced loads for both operands.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64,  'BLOCK_W': 64, 'BLOCK_J': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_W': 64, 'BLOCK_J': 64}, num_warps=8, num_stages=2),
    ],
    key=['B', 'C', 'H'],
)
@triton.jit
def triton_einsum_bchj_bhwj_bchw_kernel(
    in4_ptr,   # [B, C, H, J]  strides: (C*H*J, H*J, J, 1)
    in1_ptr,   # [B, H, W, J]  strides: (H*W*J, W*J, J, 1)
    out_ptr,   # [B, C, H, W]  strides: (C*H*W, H*W, W, 1)
    B, C, H, W, J,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,   # == W (always 64)
    BLOCK_J: tl.constexpr,   # == J (always 64)
):
    pid_c  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh  % H

    c_start   = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]
    w_offsets = tl.arange(0, BLOCK_W)              # [BLOCK_W]  covers full W
    j_offsets = tl.arange(0, BLOCK_J)              # [BLOCK_J]  covers full J

    c_mask = c_offsets < C

    # ---- Load A = in_4[b, c_range, h, :]: shape [BLOCK_C, BLOCK_J] ----
    # in_4[b,c,h,j]  at flat offset: b*C*H*J + c*H*J + h*J + j
    in4_base  = b * C * H * J + h * J
    A_offsets = c_offsets[:, None] * (H * J) + j_offsets[None, :]
    A_tile    = tl.load(in4_ptr + in4_base + A_offsets,
                        mask=c_mask[:, None], other=0.0)

    # ---- Load B^T = in_1[b, h, :, :] transposed: [BLOCK_J, BLOCK_W] ----
    # in_1[b,h,w,j]  at flat offset: b*H*W*J + h*W*J + w*J + j
    # B_T[j, w] = in_1[b,h,w,j]  → offset from in1_base = w*J + j
    in1_base  = b * H * W * J + h * W * J
    B_offsets = j_offsets[:, None] + w_offsets[None, :] * J  # [BLOCK_J, BLOCK_W]
    B_tile    = tl.load(in1_ptr + in1_base + B_offsets)

    # ---- tl.dot: [BLOCK_C, J] @ [J, BLOCK_W] → [BLOCK_C, BLOCK_W] ----
    acc = tl.zeros([BLOCK_C, BLOCK_W], dtype=tl.float32)
    acc = tl.dot(A_tile, B_tile, acc, out_dtype=tl.float32)

    # ---- Store output to out[b, c_range, h, :] ----
    # out[b,c,h,w]  at flat offset: b*C*H*W + c*H*W + h*W + w
    out_base    = b * C * H * W + h * W
    out_offsets = c_offsets[:, None] * (H * W) + w_offsets[None, :]

    acc_cast = acc.to(A_tile.dtype)   # cast fp32 accumulator back to input dtype
    tl.store(out_ptr + out_base + out_offsets, acc_cast, mask=c_mask[:, None])


# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_einsum_bchj_bhwj_bchw(in_4, in_1):
    """
    Triton replacement for einsum('bchj,bhwj->bchw', in_4, in_1).

    For each (b, h), computes:
        out[b, :, h, :] = in_4[b, :, h, :] @ in_1[b, h, :, :].T

    The CTA-parallel tl.dot avoids PyTorch einsum's string-parsing overhead
    and can exploit tensor cores directly via Triton's block-level GEMM.
    """
    B, C, H, J = in_4.shape
    W           = in_1.shape[2]

    out = torch.empty((B, C, H, W), dtype=in_4.dtype, device=in_4.device)

    grid = lambda meta: (
        triton.cdiv(C, meta['BLOCK_C']),
        B * H,
    )

    triton_einsum_bchj_bhwj_bchw_kernel[grid](
        in_4, in_1, out,
        B, C, H, W, J,
    )

    return out


def replacement_func():
    return triton_einsum_bchj_bhwj_bchw