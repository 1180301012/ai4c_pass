import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'OC', 'K_dim'],
)
@triton.jit
def _conv2d_maxpool_7x7_s2_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, H_in, W_in,
    OC, H_out, W_out, M,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    K_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Fused convolution + max_pool2d (3x3, stride=2, pad=1, dil=1).
    Each program handles a [BLOCK_M, BLOCK_N] tile of the output.
    Inner dimension K = IC * KH * KW is tiling via K_block (loop unrolled).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    m_mask = m_offs < M
    n_mask = n_offs < OC

    # Decode (batch, oh, ow) from flat output index m
    OHW = H_out * W_out
    b_idx  = m_offs // OHW            # [BLOCK_M]
    rem    = m_offs % OHW
    oh_idx = rem // W_out             # [BLOCK_M]
    ow_idx = rem % W_out              # [BLOCK_M]

    # Accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Unroll over input channels and kernel spatial dims
    for ic in range(IC):
        for kh in range(KH):
            ih    = oh_idx * stride_h + kh - pad_h         # [BLOCK_M]
            ih_ok = (ih >= 0) & (ih < H_in)

            for kw in range(KW):
                iw    = ow_idx * stride_w + kw - pad_w     # [BLOCK_M]
                iw_ok = (iw >= 0) & (iw < W_in)     # [BLOCK_M]

                valid = ih_ok & iw_ok & m_mask  # [BLOCK_M]

                # x linear index: b * IC*H_in*W_in + ic * H_in*W_in + ih * W_in + iw
                x_idx = (b_idx * IC + ic) * H_in * W_in + ih * W_in + iw  # [BLOCK_M]

                # Load input tile: [BLOCK_M]
                x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)  # [BLOCK_M]

                # weight index: (oc * IC*KH*KW + ic*KH*KW + kh*KW + kw)
                w_k = ic * KH * KW + kh * KW + kw    # scalar

                # Load weight tile: [BLOCK_N] -> broadcast to [BLOCK_M, BLOCK_N]
                w_idx = n_offs * K_dim + w_k          # [BLOCK_N]
                w_val = tl.load(w_ptr + w_idx, mask=n_mask, other=0.0)  # [BLOCK_N]

                # Outer-product accumulation: [BLOCK_M, BLOCK_N]
                acc += x_val[:, None].to(tl.float32) * w_val[None, :].to(tl.float32)

    # Now apply max_pool2d (3x3, stride=2, pad=1, dil=1) on acc
    # Output positions for each (oh, ow) in this tile
    pool_kH = 3
    pool_kW = 3
    pool_pad = 1

    # Compute output spatial indices for this tile
    oh_mat = oh_idx[:, None] + oh_offs[None, :]   # [BLOCK_M, 3]
    ow_mat = ow_idx[:, None] + ow_offs[None, :]   # [BLOCK_M, 3]
    oc_mat = n_offs[None, :]                       # [1, BLOCK_N]

    pool_valid = (oh_mat >= 0) & (oh_mat < H_out) & (ow_mat >= 0) & (ow_mat < W_out) & m_mask[:, None] & n_mask[None, :]

    pool_flat = (b_idx[:, None] * OC + oc_mat) * H_out * W_out + oh_mat * W_out + ow_mat

    pool_vals = tl.load(out_ptr + pool_flat, mask=pool_valid, other=-1e9)

    result = tl.max(pool_vals, axis=(1, 2))  # [BLOCK_M]

    # Store fused result
    out_idx = (b_idx * OC + n_offs) * H_out * W_out + oh_idx * W_out + ow_idx
    tl.store(out_ptr + out_idx, result.to(out_ptr.dtype.element_ty), mask=m_mask & n_mask)


@torch.fx.wrap
def fused_conv2d_maxpool_7x7_s2(weight, x):
    """
    Fused conv2d (7x7, stride=2, pad=3, dil=1, groups=1) + max_pool2d (3x3, stride=2, pad=1).
    weight: [OC, IC, 7, 7]
    x:      [N, IC, H_in, W_in]
    returns: [N, OC, H_out, W_out]  (H_out=(H_in+2*3-7)//2+1, W_out similarly)
    """
    OC, IC, KH, KW = weight.shape
    N, _IC, H_in, W_in = x.shape

    stride_h, stride_w = 2, 2
    pad_h,    pad_w    = 3, 3

    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1
    M     = N * H_out * W_out

    out = torch.empty((N, OC, H_out, W_out), dtype=x.dtype, device=x.device)

    # Move weights to GPU if needed
    w = weight.to(x.device)

    # Flatten weight to [OC, IC*KH*KW] for index calculation
    # (no .view() needed; use flat linear indexing in kernel)
    N_block = 32
    K_dim   = IC * KH * KW   # total inner dim (not power-of-2; padded inside loop)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    _conv2d_maxpool_7x7_s2_kernel[grid](
        x, w, out,
        N, IC, H_in, W_in,
        OC, H_out, W_out, M,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        K_dim,
    )

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pattern / replacement hooks
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_3 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_conv2d_maxpool_7x7_s2