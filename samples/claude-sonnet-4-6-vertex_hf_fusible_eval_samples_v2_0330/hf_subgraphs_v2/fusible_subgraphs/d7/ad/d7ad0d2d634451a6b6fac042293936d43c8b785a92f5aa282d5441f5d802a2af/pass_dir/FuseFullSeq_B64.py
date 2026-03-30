"""
FuseFullSeq_B64.py

Matches: linear → permute → reshape(64, -1, 16, 16) → interpolate(128×128)
for batch-size-64 graphs (float32/8 and float16/8).

Fuses the whole pipeline into two Triton kernels:
  1. GEMM kernel   : writes [B, C, S] in the transposed layout directly
  2. Bilinear kernel: 16×16 → 128×128 upsampling
"""

import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.permute(0, 2, 1)
    tmp_4 = tmp_3.reshape(64, -1, 16, 16)
    tmp_5 = torch.nn.functional.interpolate(
        tmp_4, size=(128, 128), mode='bilinear', align_corners=False
    )
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Kernel 1: fused linear + transposed write
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64,  'BLOCK_S': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 128, 'BLOCK_K': 32},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 64,  'BLOCK_S': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
    ],
    key=['B', 'S', 'K', 'C'],
)
@triton.jit
def gemm_permute_kernel_b64(
    in2_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, K, C,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)

    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    c_mask = c_off < C
    s_mask = s_off < S

    acc = tl.zeros([BLOCK_C, BLOCK_S], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        w = tl.load(w_ptr + c_off[:, None] * K + k_off[None, :],
                    mask=c_mask[:, None] & k_mask[None, :], other=0.0)
        in2 = tl.load(in2_ptr + pid_b * S * K + s_off[:, None] * K + k_off[None, :],
                      mask=s_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(w, tl.trans(in2), allow_tf32=True)

    bias = tl.load(bias_ptr + c_off, mask=c_mask, other=0.0).to(tl.float32)
    acc += bias[:, None]

    tl.store(out_ptr + pid_b * C * S + c_off[:, None] * S + s_off[None, :],
             acc, mask=c_mask[:, None] & s_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel 2: bilinear upsampling 16×16 → 128×128
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8},  num_warps=4),
        triton.Config({'BLOCK_H': 16}, num_warps=4),
        triton.Config({'BLOCK_H': 32}, num_warps=8),
        triton.Config({'BLOCK_H': 16}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def bilinear_16to128_kernel_b64(
    inp_ptr, out_ptr,
    N,
    H_IN:  tl.constexpr,
    W_IN:  tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_range = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_range = tl.arange(0, W_OUT)
    h_mask  = h_range < H_OUT

    scale = H_IN / H_OUT

    h_in_f = -0.5 + (h_range.to(tl.float32) + 0.5) * scale
    h0_f   = tl.floor(h_in_f)
    h0     = h0_f.to(tl.int32)
    lh     = h_in_f - h0_f
    h0c    = tl.maximum(0, tl.minimum(H_IN - 1, h0))
    h1c    = tl.maximum(0, tl.minimum(H_IN - 1, h0 + 1))

    w_in_f = -0.5 + (w_range.to(tl.float32) + 0.5) * scale
    w0_f   = tl.floor(w_in_f)
    w0     = w0_f.to(tl.int32)
    lw     = w_in_f - w0_f
    w0c    = tl.maximum(0, tl.minimum(W_IN - 1, w0))
    w1c    = tl.maximum(0, tl.minimum(W_IN - 1, w0 + 1))

    base    = pid_n * H_IN * W_IN
    hmask2d = h_mask[:, None]

    v00 = tl.load(inp_ptr + base + h0c[:, None] * W_IN + w0c[None, :],
                  mask=hmask2d, other=0.0).to(tl.float32)
    v01 = tl.load(inp_ptr + base + h0c[:, None] * W_IN + w1c[None, :],
                  mask=hmask2d, other=0.0).to(tl.float32)
    v10 = tl.load(inp_ptr + base + h1c[:, None] * W_IN + w0c[None, :],
                  mask=hmask2d, other=0.0).to(tl.float32)
    v11 = tl.load(inp_ptr + base + h1c[:, None] * W_IN + w1c[None, :],
                  mask=hmask2d, other=0.0).to(tl.float32)

    lh2 = lh[:, None]
    lw2 = lw[None, :]
    result = (
        (1.0 - lh2) * (1.0 - lw2) * v00 +
        (1.0 - lh2) *        lw2  * v01 +
               lh2  * (1.0 - lw2) * v10 +
               lh2  *        lw2  * v11
    )

    out_idx = pid_n * H_OUT * W_OUT + h_range[:, None] * W_OUT + w_range[None, :]
    tl.store(out_ptr + out_idx, result, mask=hmask2d)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_seq_b64(in_0, in_1, in_2):
    B = in_2.shape[0]   # 64
    S = in_2.shape[1]   # 256
    K = in_2.shape[2]   # 512
    C = in_1.shape[0]   # 768
    N = B * C
    H_in = W_in = 16
    H_out = W_out = 128

    orig_dtype = in_2.dtype
    in_0_gpu = in_0.to(device=in_2.device, dtype=orig_dtype)
    in_1_gpu = in_1.to(device=in_2.device, dtype=orig_dtype)
    in_2_c   = in_2.contiguous()

    # Step 1: GEMM → [B, C, S] float32
    gemm_out = torch.empty((B, C, S), dtype=torch.float32, device=in_2.device)
    grid1 = lambda meta: (B, triton.cdiv(C, meta['BLOCK_C']), triton.cdiv(S, meta['BLOCK_S']))
    gemm_permute_kernel_b64[grid1](in_2_c, in_1_gpu, in_0_gpu, gemm_out, B, S, K, C)

    # View as [N, 16, 16] for the bilinear kernel
    gemm_4d = gemm_out.view(N, H_in, W_in)

    # Step 2: bilinear → [N, 128, 128] float32
    interp_out = torch.empty((N, H_out, W_out), dtype=torch.float32, device=in_2.device)
    grid2 = lambda meta: (N, triton.cdiv(H_out, meta['BLOCK_H']))
    bilinear_16to128_kernel_b64[grid2](gemm_4d, interp_out, N,
                                       H_IN=H_in, W_IN=W_in,
                                       H_OUT=H_out, W_OUT=W_out)

    # Reshape to [B, C, 128, 128] and cast to original dtype
    out = interp_out.view(B, C, H_out, W_out)
    if orig_dtype != torch.float32:
        out = out.to(orig_dtype)
    return out


def replacement_func():
    return fused_seq_b64