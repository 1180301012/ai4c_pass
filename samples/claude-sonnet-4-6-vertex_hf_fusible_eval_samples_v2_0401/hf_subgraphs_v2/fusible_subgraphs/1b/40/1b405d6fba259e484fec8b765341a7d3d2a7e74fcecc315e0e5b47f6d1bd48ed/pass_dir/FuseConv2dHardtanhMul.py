import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    """
    Match: 1x1 conv2d  +  hardtanh(0,6) on in_3  +  elementwise multiply.
    Positional args mirror model.py exactly.
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ── Triton kernel: fully fused 1x1-conv + hardtanh(0,6) + elemwise-mul ───────
#
# Computation:
#   output[n, c_out, h, w] = clamp(in3[n,c_out,h,w], 0,6)
#                            * (bias[c_out] + Σ_k in2[n,k,h,w]*weight[c_out,k])
#
# Tiling: M = N*H*W  (batch × spatial),  K = C_in (24),  N_out = C_out (96)
# We tile M (pid_m) and C_out (pid_n), with BLOCK_K=32 (≥ C_in=24) for tl.dot.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8,  num_stages=2),
    ],
    key=['M_total', 'C_out'],
)
@triton.jit
def _fused_1x1conv_hardtanh_mul_kernel(
    in2_ptr,    # [N, C_in, H, W]  NCHW
    wgt_ptr,    # [C_out, C_in]    (reshaped from [C_out, C_in, 1, 1])
    bias_ptr,   # [C_out]
    in3_ptr,    # [N, C_out, H, W] NCHW
    out_ptr,    # [N, C_out, H, W] NCHW  (output)
    N,          # batch size
    C_in,       # input channels  (24)
    C_out,      # output channels (96)
    HW,         # H * W
    M_total,    # N * H * W
    BLOCK_M: tl.constexpr,   # tile rows   (spatial batch)
    BLOCK_N: tl.constexpr,   # tile cols   (output channels)
    BLOCK_K: tl.constexpr,   # = 32  (≥ C_in, power-of-2 for tl.dot)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ── row / column offsets ──────────────────────────────────────────────
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # spatial positions
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # output channels
    k_offs = tl.arange(0, BLOCK_K)                      # input channels (padded)

    m_mask = m_offs < M_total
    n_mask = n_offs < C_out
    k_mask = k_offs < C_in

    # Decompose flat spatial index → batch index + spatial index
    n_idx  = m_offs // HW    # [BLOCK_M]  batch index
    hw_idx = m_offs % HW     # [BLOCK_M]  h*W + w

    # ── Load in2 tile  [BLOCK_M, BLOCK_K] ────────────────────────────────
    # in2[n, k, h, w]  at  n_idx*C_in*HW + k*HW + hw_idx
    in2_ptrs = (in2_ptr
                + n_idx[:, None] * (C_in * HW)
                + k_offs[None, :] * HW
                + hw_idx[:, None])
    in2_tile = tl.load(in2_ptrs,
                       mask=m_mask[:, None] & k_mask[None, :],
                       other=0.0)   # [BLOCK_M, BLOCK_K]

    # ── Load weight tile  [BLOCK_K, BLOCK_N] ─────────────────────────────
    # weight[c_out, c_in] stored contiguously as [C_out*C_in]
    # We need weight.T[k, n] = weight[n_offs[n], k_offs[k]]
    #   address = wgt_ptr + n_offs[n]*C_in + k_offs[k]
    wgt_ptrs = (wgt_ptr
                + k_offs[:, None]
                + n_offs[None, :] * C_in)
    wgt_tile = tl.load(wgt_ptrs,
                       mask=k_mask[:, None] & n_mask[None, :],
                       other=0.0)   # [BLOCK_K, BLOCK_N]

    # ── Matrix multiply: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] ─────────
    acc = tl.dot(in2_tile, wgt_tile, out_dtype=tl.float32)   # [BLOCK_M, BLOCK_N]

    # ── Add bias [BLOCK_N] ────────────────────────────────────────────────
    bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # ── Load in3 tile  [BLOCK_M, BLOCK_N] ────────────────────────────────
    # in3[n, c_out, h, w]  at  n_idx*C_out*HW + c_out*HW + hw_idx
    in3_ptrs = (in3_ptr
                + n_idx[:, None] * (C_out * HW)
                + n_offs[None, :] * HW
                + hw_idx[:, None])
    in3_tile = tl.load(in3_ptrs,
                       mask=m_mask[:, None] & n_mask[None, :],
                       other=0.0).to(tl.float32)   # [BLOCK_M, BLOCK_N]

    # ── Fused hardtanh(0, 6)  +  elementwise multiply ────────────────────
    clamped = tl.minimum(tl.maximum(in3_tile, 0.0), 6.0)
    result  = clamped * acc

    # ── Store output (same NCHW layout as in3) ────────────────────────────
    out_ptrs = (out_ptr
                + n_idx[:, None] * (C_out * HW)
                + n_offs[None, :] * HW
                + hw_idx[:, None])
    tl.store(out_ptrs,
             result.to(in2_tile.dtype),
             mask=m_mask[:, None] & n_mask[None, :])


# ── Replacement wrapper ───────────────────────────────────────────────────────
@torch.fx.wrap
def fused_conv_hardtanh_mul(bias, weight, x, y):
    """
    Fully Triton-fused replacement for:
        conv2d(x, weight, bias, stride=1, pad=0, dil=1, groups=1)  [1×1 kernel]
        hardtanh(y, 0, 6)
        result = hardtanh_out * conv_out
    Arguments: (in_0=bias, in_1=weight, in_2=x, in_3=y)
    """
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    HW    = H * W
    M_total = N * HW

    # Move CPU weights to GPU; flatten 1×1 kernel dims
    dev = x.device
    weight_2d = weight.to(dev).reshape(C_out, C_in).contiguous()
    bias_dev  = bias.to(dev)

    x_c = x.contiguous()
    y_c = y.contiguous()
    out = torch.empty((N, C_out, H, W), dtype=x.dtype, device=dev)

    BLOCK_K = 32   # ≥ C_in=24, power-of-2 for tl.dot

    grid = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(C_out,   meta['BLOCK_N']),
    )

    _fused_1x1conv_hardtanh_mul_kernel[grid](
        x_c, weight_2d, bias_dev, y_c, out,
        N, C_in, C_out, HW, M_total,
        BLOCK_K=BLOCK_K,
    )

    return out


# ── Required entry-point ──────────────────────────────────────────────────────
def replacement_func():
    return fused_conv_hardtanh_mul