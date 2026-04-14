"""
Pass: Fuse conv2d(in_2,...) -> stack([x],0) -> sum(0) -> cat([result,in_3],1)
into a single fused Triton kernel (1x1 GEMM + cat).

stack([x], dim=0).sum(dim=0) is a no-op (identity), so the real work is:
  conv2d(in_2, weight, bias)  followed by  cat([result, in_3], dim=1)

We fuse by:
  1. Pre-allocating the cat output [N, C_out+C_y, H, W]
  2. Running a Triton GEMM kernel to write conv result into first C_out channels
  3. Running a Triton copy kernel to write in_3 into remaining C_y channels
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern (must mirror model.py dataflow exactly)
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.stack([conv2d], dim=0)
    tmp_4 = tmp_3.sum(dim=0)
    tmp_5 = torch.cat([tmp_4, in_3], 1)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    # (bias, weight, conv_input, cat_other)
    return (in_0, in_1, in_2, in_3)


# ─────────────────────────────────────────────────────────────────────────────
# Triton Kernels
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['N_M', 'C_out', 'C_in'],
)
@triton.jit
def _conv1x1_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N_M, C_in, C_out, C_total, HW,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """GEMM for 1x1 conv (NCHW) writing directly into the cat output buffer."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    mask_m = m_offs < N_M
    mask_n = n_offs < C_out

    n_idx  = m_offs // HW
    hw_idx = m_offs % HW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offs < C_in

        # Load x tile [BLOCK_M, BLOCK_K]
        # x NCHW: x[n, k, hw] = x_ptr + n*C_in*HW + k*HW + hw
        x_idx = (n_idx[:, None] * (C_in * HW)
                 + k_offs[None, :] * HW
                 + hw_idx[:, None])
        x_tile = tl.load(x_ptr + x_idx,
                         mask=mask_m[:, None] & mask_k[None, :],
                         other=0.0).to(tl.float32)

        # Load w tile [BLOCK_K, BLOCK_N]
        # w [C_out, C_in]: w^T[k, n] = w[n, k] = w_ptr + n*C_in + k
        w_idx = (n_offs[None, :] * C_in
                 + k_offs[:, None])
        w_tile = tl.load(w_ptr + w_idx,
                         mask=mask_k[:, None] & mask_n[None, :],
                         other=0.0).to(tl.float32)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    b_vals = tl.load(b_ptr + n_offs, mask=mask_n, other=0.0).to(tl.float32)
    acc += b_vals[None, :]

    # Store into the cat output buffer (channel stride = HW, batch stride = C_total*HW)
    out_idx = (n_idx[:, None] * (C_total * HW)
               + n_offs[None, :] * HW
               + hw_idx[:, None])

    if IS_FP16:
        tl.store(out_ptr + out_idx, acc.to(tl.float16),
                 mask=mask_m[:, None] & mask_n[None, :])
    elif IS_BF16:
        tl.store(out_ptr + out_idx, acc.to(tl.bfloat16),
                 mask=mask_m[:, None] & mask_n[None, :])
    else:
        tl.store(out_ptr + out_idx, acc,
                 mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_cat_kernel(
    src_ptr, dst_ptr,
    N, C_src, C_dst_off, C_total, HW,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy src [N, C_src, H*W] into dst[n, C_dst_off:C_dst_off+C_src, h*w].
    Uses a large BLOCK_SIZE for high memory throughput.
    """
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C_src * HW
    mask  = offs < total

    # Decode flat index into (n, c, hw)
    tmp = offs
    hw  = tmp % HW
    tmp = tmp // HW
    c   = tmp % C_src
    n   = tmp // C_src

    src_idx = n * (C_src * HW) + c * HW + hw
    dst_idx = n * (C_total * HW) + (C_dst_off + c) * HW + hw

    vals = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + dst_idx, vals, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def _fused_conv1x1_cat_a(bias, weight, conv_input, cat_other):
    """
    bias:       [C_out]
    weight:     [C_out, C_in, 1, 1]
    conv_input: [N, C_in, H, W]
    cat_other:  [N, C_y,  H, W]
    returns:    [N, C_out + C_y, H, W]
    """
    N, C_in, H, W = conv_input.shape
    C_out = weight.shape[0]
    C_y   = cat_other.shape[1]
    C_total = C_out + C_y
    HW    = H * W
    N_M   = N * HW

    # Allocate the full cat output
    out = torch.empty((N, C_total, H, W),
                      dtype=conv_input.dtype,
                      device=conv_input.device)

    is_fp16  = conv_input.dtype == torch.float16
    is_bf16  = conv_input.dtype == torch.bfloat16

    # ── 1.  GEMM: conv1x1 → first C_out channels ──────────────────────────
    # weight [C_out, C_in, 1, 1] has stride (C_in,1,1,1); same layout as [C_out,C_in]
    # Use lambda grid so it adapts to the BLOCK_M/BLOCK_N chosen by autotune
    _nm, _co = N_M, C_out
    grid_gemm = lambda meta: (triton.cdiv(_nm, meta['BLOCK_M']),
                              triton.cdiv(_co, meta['BLOCK_N']))
    _conv1x1_gemm_kernel[grid_gemm](
        conv_input, weight, bias, out,
        N_M, C_in, C_out, C_total, HW,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )

    # ── 2.  Copy: cat_other → last C_y channels ────────────────────────────
    COPY_BS  = 2048
    copy_grid = (triton.cdiv(N * C_y * HW, COPY_BS),)
    _copy_cat_kernel[copy_grid](
        cat_other, out,
        N, C_y, C_out, C_total, HW,
        BLOCK_SIZE=COPY_BS,
    )

    return out


def replacement_func():
    return _fused_conv1x1_cat_a