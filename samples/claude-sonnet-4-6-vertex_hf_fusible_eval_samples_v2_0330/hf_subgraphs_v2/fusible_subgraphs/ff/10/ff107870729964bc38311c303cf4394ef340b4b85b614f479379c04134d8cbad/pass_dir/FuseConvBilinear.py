"""
Pass: FuseConvBilinear (fallback)

Matches just:  conv2d(1x1) → bilinear_interpolate(size=(512,512))  → tmp_11

Fallback for when the full-graph dead-code-elimination passes did not match.
Works for both float16 and bfloat16.
Uses pure Triton (Triton GEMM + Triton bilinear upsample).
"""

import torch
import triton
import triton.language as tl


# ── GEMM kernel, float16 output ──

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=5, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_f16_fb_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k  = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk  + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_am_true = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    bias = tl.load(bias_ptr + offs_am_true, mask=offs_am_true < M, other=0.0).to(tl.float32)
    acc += bias[:, None]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs  = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask  = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# ── GEMM kernel, bfloat16 output ──

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=5, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_bf16_fb_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k  = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk  + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_am_true = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    bias = tl.load(bias_ptr + offs_am_true, mask=offs_am_true < M, other=0.0).to(tl.float32)
    acc += bias[:, None]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs  = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask  = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


# ── Bilinear upsample kernel, float16 output ──

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['N', 'C', 'H_in', 'W_in', 'H_out', 'W_out'],
)
@triton.jit
def bilinear_upsample_f16_fb_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    scale_h, scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total   = N * C * H_out * W_out
    mask    = offsets < total

    x_out = offsets % W_out
    y_out = (offsets // W_out) % H_out
    c_idx = (offsets // (H_out * W_out)) % C
    n_idx = offsets // (C * H_out * W_out)

    y_src = (y_out.to(tl.float32) + 0.5) * scale_h - 0.5
    x_src = (x_out.to(tl.float32) + 0.5) * scale_w - 0.5

    y0_f = tl.floor(y_src);  x0_f = tl.floor(x_src)
    y0 = y0_f.to(tl.int32);  x0 = x0_f.to(tl.int32)
    y1 = y0 + 1;              x1 = x0 + 1

    y0_c = tl.minimum(tl.maximum(y0, 0), H_in - 1)
    y1_c = tl.minimum(tl.maximum(y1, 0), H_in - 1)
    x0_c = tl.minimum(tl.maximum(x0, 0), W_in - 1)
    x1_c = tl.minimum(tl.maximum(x1, 0), W_in - 1)

    wy1 = y_src - y0_f;  wx1 = x_src - x0_f
    wy0 = 1.0 - wy1;     wx0 = 1.0 - wx1

    base  = n_idx * (C * H_in * W_in) + c_idx * (H_in * W_in)
    idx00 = base + y0_c * W_in + x0_c
    idx01 = base + y0_c * W_in + x1_c
    idx10 = base + y1_c * W_in + x0_c
    idx11 = base + y1_c * W_in + x1_c

    v00 = tl.load(x_ptr + idx00, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(x_ptr + idx01, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(x_ptr + idx10, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(x_ptr + idx11, mask=mask, other=0.0).to(tl.float32)

    result = v00 * wy0 * wx0 + v01 * wy0 * wx1 + v10 * wy1 * wx0 + v11 * wy1 * wx1
    tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)


# ── Bilinear upsample kernel, bfloat16 output ──

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['N', 'C', 'H_in', 'W_in', 'H_out', 'W_out'],
)
@triton.jit
def bilinear_upsample_bf16_fb_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    scale_h, scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total   = N * C * H_out * W_out
    mask    = offsets < total

    x_out = offsets % W_out
    y_out = (offsets // W_out) % H_out
    c_idx = (offsets // (H_out * W_out)) % C
    n_idx = offsets // (C * H_out * W_out)

    y_src = (y_out.to(tl.float32) + 0.5) * scale_h - 0.5
    x_src = (x_out.to(tl.float32) + 0.5) * scale_w - 0.5

    y0_f = tl.floor(y_src);  x0_f = tl.floor(x_src)
    y0 = y0_f.to(tl.int32);  x0 = x0_f.to(tl.int32)
    y1 = y0 + 1;              x1 = x0 + 1

    y0_c = tl.minimum(tl.maximum(y0, 0), H_in - 1)
    y1_c = tl.minimum(tl.maximum(y1, 0), H_in - 1)
    x0_c = tl.minimum(tl.maximum(x0, 0), W_in - 1)
    x1_c = tl.minimum(tl.maximum(x1, 0), W_in - 1)

    wy1 = y_src - y0_f;  wx1 = x_src - x0_f
    wy0 = 1.0 - wy1;     wx0 = 1.0 - wx1

    base  = n_idx * (C * H_in * W_in) + c_idx * (H_in * W_in)
    idx00 = base + y0_c * W_in + x0_c
    idx01 = base + y0_c * W_in + x1_c
    idx10 = base + y1_c * W_in + x0_c
    idx11 = base + y1_c * W_in + x1_c

    v00 = tl.load(x_ptr + idx00, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(x_ptr + idx01, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(x_ptr + idx10, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(x_ptr + idx11, mask=mask, other=0.0).to(tl.float32)

    result = v00 * wy0 * wx0 + v01 * wy0 * wx1 + v10 * wy1 * wx0 + v11 * wy1 * wx1
    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=mask)


# ── Python wrapper (dtype-aware) ──

@torch.fx.wrap
def triton_conv_bilinear_fallback(in_10, in_8, in_7):
    """
    Pure-Triton 1×1 conv + bilinear upsample.  Handles float16 and bfloat16.
    """
    device = in_10.device
    dtype  = in_10.dtype

    weight = in_8.to(device=device, dtype=dtype)
    bias   = in_7.to(device=device, dtype=dtype)

    N, Cin, H, W = in_10.shape
    Cout = weight.shape[0]
    HW   = H * W

    A = weight.view(Cout, Cin)
    B = in_10.view(Cin, HW)
    C = torch.empty(Cout, HW, dtype=dtype, device=device)

    def gemm_grid(meta):
        return (triton.cdiv(Cout, meta['BLOCK_M']) * triton.cdiv(HW, meta['BLOCK_N']),)

    if dtype == torch.float16:
        gemm_bias_f16_fb_kernel[gemm_grid](
            A, B, bias, C,
            Cout, HW, Cin,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
    else:
        gemm_bias_bf16_fb_kernel[gemm_grid](
            A, B, bias, C,
            Cout, HW, Cin,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )

    conv_out = C.view(N, Cout, H, W)

    H_out, W_out = 512, 512
    total  = N * Cout * H_out * W_out
    out    = torch.empty(N, Cout, H_out, W_out, dtype=dtype, device=device)
    scale_h = float(H) / float(H_out)
    scale_w = float(W) / float(W_out)

    def up_grid(meta):
        return ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    if dtype == torch.float16:
        bilinear_upsample_f16_fb_kernel[up_grid](
            conv_out, out,
            N, Cout, H, W, H_out, W_out,
            scale_h, scale_w,
        )
    else:
        bilinear_upsample_bf16_fb_kernel[up_grid](
            conv_out, out,
            N, Cout, H, W, H_out, W_out,
            scale_h, scale_w,
        )

    return out


@torch.fx.wrap
def triton_bilinear_upsample_only(x):
    """
    Triton bilinear upsample: any input → 512×512 output.
    Handles float16 and bfloat16.
    """
    device = x.device
    dtype  = x.dtype
    N, C, H_in, W_in = x.shape
    H_out, W_out = 512, 512
    total = N * C * H_out * W_out
    out = torch.empty(N, C, H_out, W_out, dtype=dtype, device=device)
    scale_h = float(H_in) / float(H_out)
    scale_w = float(W_in) / float(W_out)

    def up_grid(meta):
        return ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    if dtype == torch.float16:
        bilinear_upsample_f16_fb_kernel[up_grid](
            x, out,
            N, C, H_in, W_in, H_out, W_out,
            scale_h, scale_w,
        )
    else:
        bilinear_upsample_bf16_fb_kernel[up_grid](
            x, out,
            N, C, H_in, W_in, H_out, W_out,
            scale_h, scale_w,
        )
    return out


@torch.fx.wrap
def triton_conv_only(x, w, b):
    """
    1×1 conv via Triton GEMM + Triton fused bias-add.
    Returns [N, Cout, H, W] — no upsampling, so downstream PyTorch
    interpolate can still consume the output correctly.
    """
    device = x.device
    dtype  = x.dtype
    w_dev = w.to(device=device, dtype=dtype)
    b_dev = b.to(device=device, dtype=dtype)
    N, Cin, H, W = x.shape
    Cout = w_dev.shape[0]
    HW   = H * W

    A = w_dev.view(Cout, Cin)   # [Cout, Cin]
    B = x.view(Cin, HW)          # [Cin, HW]
    C = torch.empty(Cout, HW, dtype=dtype, device=device)

    def gemm_grid(meta):
        return (triton.cdiv(Cout, meta['BLOCK_M']) * triton.cdiv(HW, meta['BLOCK_N']),)

    if dtype == torch.float16:
        gemm_bias_f16_fb_kernel[gemm_grid](
            A, B, b_dev, C,
            Cout, HW, Cin,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
    else:
        gemm_bias_bf16_fb_kernel[gemm_grid](
            A, B, b_dev, C,
            Cout, HW, Cin,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )

    return C.view(N, Cout, H, W)


def pattern(x, w, b):
    """
    Match torch.conv2d with stride=1, pad=0, dil=1, groups=1, non-None bias.
    This matches BOTH useful (conv2d→tmp_11 path) and dead-code (conv2d_2→tmp_16 path).
    Excludes conv2d_1 (which has padding=(1,1) and bias=None).
    """
    return torch.conv2d(x, w, b, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(x, w, b):
    return (x, w, b)


def replacement_func():
    return triton_conv_only