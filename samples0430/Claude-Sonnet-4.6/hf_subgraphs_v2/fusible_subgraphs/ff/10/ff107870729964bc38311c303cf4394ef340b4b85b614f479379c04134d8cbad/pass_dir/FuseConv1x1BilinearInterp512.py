import torch
import triton
import triton.language as tl

from pass_dir.shared_kernels import _run_fused_conv1x1_bilinear


def pattern(in_10, in_8, in_7):
    """
    Match the 1x1 conv2d (Python-level leaf).
    Replacement returns [1,C_out,512,512]; subsequent F.interpolate becomes no-op.
    Both conv2d branches are matched (first branch is live, second is dead code).
    """
    conv2d = torch.conv2d(in_10, in_8, in_7, (1, 1), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_10, in_8, in_7):
    return (in_10, in_8, in_7)


@torch.fx.wrap
def dispatch_wrapper(in_10, in_8, in_7, route):
    if route == "route_default":
        return _run_fused_conv1x1_bilinear(in_10, in_8, in_7)
    elif route == "route_vec":
        return _run_fused_conv1x1_bilinear(in_10, in_8, in_7)
    elif route == "route_py":
        return _run_fused_conv1x1_bilinear(in_10, in_8, in_7)
    return _run_fused_conv1x1_bilinear(in_10, in_8, in_7)


def replacement_func():
    return dispatch_wrapper


# ─── Triton GEMM kernel ───────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64},
                      num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = acc + tl.dot(a, b, out_dtype=tl.float32)
    bias = tl.load(Bias_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    acc = acc + bias[:, None]
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ─── Triton bilinear upsample kernel ─────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
    ],
    key=['C', 'H_out', 'W_out'],
)
@triton.jit
def _bilinear_upsample_kernel(
    X_ptr, Y_ptr,
    C, H_in, W_in, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < C * H_out * W_out
    ow = offs % W_out
    oh = (offs // W_out) % H_out
    c  = offs // (H_out * W_out)
    scale_h = tl.cast(H_in, tl.float32) / tl.cast(H_out, tl.float32)
    scale_w = tl.cast(W_in, tl.float32) / tl.cast(W_out, tl.float32)
    ih_f = (tl.cast(oh, tl.float32) + 0.5) * scale_h - 0.5
    iw_f = (tl.cast(ow, tl.float32) + 0.5) * scale_w - 0.5
    ih_i = ih_f.to(tl.int32)
    ih_i = tl.where(ih_f < ih_i.to(tl.float32), ih_i - 1, ih_i)
    iw_i = iw_f.to(tl.int32)
    iw_i = tl.where(iw_f < iw_i.to(tl.float32), iw_i - 1, iw_i)
    ih0 = tl.maximum(ih_i, 0);    ih1 = tl.minimum(ih_i + 1, H_in - 1)
    iw0 = tl.maximum(iw_i, 0);    iw1 = tl.minimum(iw_i + 1, W_in - 1)
    h_frac = tl.maximum(ih_f - ih_i.to(tl.float32), 0.0)
    w_frac = tl.maximum(iw_f - iw_i.to(tl.float32), 0.0)
    w00 = (1.0 - h_frac) * (1.0 - w_frac);  w01 = (1.0 - h_frac) * w_frac
    w10 = h_frac * (1.0 - w_frac);           w11 = h_frac * w_frac
    base = c * H_in * W_in
    x00 = tl.load(X_ptr + base + ih0 * W_in + iw0, mask=mask, other=0.0)
    x01 = tl.load(X_ptr + base + ih0 * W_in + iw1, mask=mask, other=0.0)
    x10 = tl.load(X_ptr + base + ih1 * W_in + iw0, mask=mask, other=0.0)
    x11 = tl.load(X_ptr + base + ih1 * W_in + iw1, mask=mask, other=0.0)
    result = w00 * x00 + w01 * x01 + w10 * x10 + w11 * x11
    if IS_BFLOAT16:
        tl.store(Y_ptr + offs, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(Y_ptr + offs, result.to(tl.float16), mask=mask)


# ─── Python wrapper ───────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_conv1x1_bilinear_512(in_10, in_8, in_7):
    device = in_10.device
    dtype  = in_10.dtype
    C_in  = in_10.shape[1];  H_in = in_10.shape[2];  W_in = in_10.shape[3]
    C_out = in_8.shape[0];   HW   = H_in * W_in
    H_out, W_out = 512, 512
    W_mat = in_8.to(device=device).view(C_out, C_in).contiguous()
    B_vec = in_7.to(device=device).contiguous()
    X_mat = in_10.contiguous().view(C_in, HW)
    # GEMM
    intermediate = torch.empty((C_out, HW), device=device, dtype=torch.float32)
    grid_g = lambda m: (triton.cdiv(C_out, m['BLOCK_M']), triton.cdiv(HW, m['BLOCK_N']))
    _gemm_bias_kernel[grid_g](W_mat, X_mat, B_vec, intermediate, C_out, HW, C_in)
    # Bilinear upsample
    output = torch.empty((1, C_out, H_out, W_out), device=device, dtype=dtype)
    total  = C_out * H_out * W_out
    grid_u = lambda m: (triton.cdiv(total, m['BLOCK_SIZE']),)
    _bilinear_upsample_kernel[grid_u](
        intermediate.view(-1), output.view(-1),
        C_out, H_in, W_in, H_out, W_out,
        IS_BFLOAT16=(dtype == torch.bfloat16),
    )
    return output


def replacement_func():
    return fused_conv1x1_bilinear_512



# ──────────────────────────────────────────────────────────────────────────────
# Legacy (unused) stubs kept below to avoid import errors from other modules
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64},
                      num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_fp32out_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    C[M, N] = A[M, K] @ B[K, N] + Bias[M]
    A and B can be float16 or bfloat16; output C is float32.
    A is contiguous row-major [M, K], B is [K, N], C is [M, N].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=a_mask, other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=b_mask, other=0.0)

        acc = acc + tl.dot(a, b, out_dtype=tl.float32)

    bias = tl.load(Bias_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    acc = acc + bias[:, None]

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=c_mask)


# ─── Kernel 2: Bilinear upsample ─────────────────────────────────────────────
# Reads float32 intermediate, writes float16 or bfloat16 output.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['C', 'H_out', 'W_out'],
)
@triton.jit
def bilinear_upsample_kernel(
    X_ptr, Y_ptr,
    C, H_in, W_in, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
):
    """
    Bilinear upsample (align_corners=False).
    X: float32 flat [C * H_in * W_in]
    Y: float16 or bfloat16 flat [C * H_out * W_out]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = C * H_out * W_out
    mask = offs < total

    # Decode output flat index → (c, oh, ow)
    ow = offs % W_out
    tmp = offs // W_out
    oh = tmp % H_out
    c  = tmp // H_out

    # Bilinear source coordinates (align_corners=False)
    scale_h = tl.cast(H_in, tl.float32) / tl.cast(H_out, tl.float32)
    scale_w = tl.cast(W_in, tl.float32) / tl.cast(W_out, tl.float32)

    ih_f = (tl.cast(oh, tl.float32) + 0.5) * scale_h - 0.5
    iw_f = (tl.cast(ow, tl.float32) + 0.5) * scale_w - 0.5

    # Floor (correct for negative values too)
    ih_i = ih_f.to(tl.int32)
    ih_i = tl.where(ih_f < ih_i.to(tl.float32), ih_i - 1, ih_i)
    iw_i = iw_f.to(tl.int32)
    iw_i = tl.where(iw_f < iw_i.to(tl.float32), iw_i - 1, iw_i)

    # Clamp integer coordinates to valid range [0, H_in-1] / [0, W_in-1]
    ih0 = tl.maximum(ih_i,     0)
    ih1 = tl.minimum(ih_i + 1, H_in - 1)
    iw0 = tl.maximum(iw_i,     0)
    iw1 = tl.minimum(iw_i + 1, W_in - 1)

    # Fractional parts (clamped to ≥ 0 to handle border)
    h_frac = tl.maximum(ih_f - ih_i.to(tl.float32), 0.0)
    w_frac = tl.maximum(iw_f - iw_i.to(tl.float32), 0.0)

    w00 = (1.0 - h_frac) * (1.0 - w_frac)
    w01 = (1.0 - h_frac) * w_frac
    w10 = h_frac * (1.0 - w_frac)
    w11 = h_frac * w_frac

    base = c * H_in * W_in
    x00 = tl.load(X_ptr + base + ih0 * W_in + iw0, mask=mask, other=0.0)
    x01 = tl.load(X_ptr + base + ih0 * W_in + iw1, mask=mask, other=0.0)
    x10 = tl.load(X_ptr + base + ih1 * W_in + iw0, mask=mask, other=0.0)
    x11 = tl.load(X_ptr + base + ih1 * W_in + iw1, mask=mask, other=0.0)

    result = w00 * x00 + w01 * x01 + w10 * x10 + w11 * x11

    if IS_BFLOAT16:
        tl.store(Y_ptr + offs, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(Y_ptr + offs, result.to(tl.float16), mask=mask)


# ─── Python wrapper ──────────────────────────────────────────────────────────

@torch.fx.wrap
def conv1x1_bilinear_upsample_512(in_10, in_8, in_7):
    """
    Fused 1x1 conv + bilinear upsample to 512×512.

    in_10 : [1, C_in, H_in, W_in]  — input feature map (on GPU)
    in_8  : [C_out, C_in, 1, 1]    — conv weight (may be on CPU)
    in_7  : [C_out]                 — conv bias   (may be on CPU)
    Returns: [1, C_out, 512, 512]
    """
    device = in_10.device
    dtype  = in_10.dtype

    C_in  = in_10.shape[1]
    H_in  = in_10.shape[2]
    W_in  = in_10.shape[3]
    C_out = in_8.shape[0]
    HW    = H_in * W_in
    H_out = 512
    W_out = 512

    # ── Step 1: GEMM (pass raw tensors — no .view/.to/.contiguous) ────────────
    # in_8 [C_out, C_in, 1, 1] → treated as [C_out, C_in]: strides [C_in, 1]
    # in_10 [1, C_in, H_in, W_in] → treated as [C_in, HW]:  strides [HW, 1]
    intermediate = torch.empty((C_out, HW), device=device, dtype=torch.float32)

    grid_gemm = lambda meta: (
        triton.cdiv(C_out, meta['BLOCK_M']),
        triton.cdiv(HW,    meta['BLOCK_N']),
    )
    gemm_bias_fp32out_kernel[grid_gemm](
        in_8, in_10, in_7, intermediate,
        C_out, HW, C_in,
    )

    # ── Step 2: Bilinear 4× upsample ─────────────────────────────────────────
    output = torch.empty((1, C_out, H_out, W_out), device=device, dtype=dtype)
    total  = C_out * H_out * W_out

    is_bf16 = (dtype == torch.bfloat16)

    grid_up = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    bilinear_upsample_kernel[grid_up](
        intermediate,
        output,
        C_out, H_in, W_in, H_out, W_out,
        IS_BFLOAT16=is_bf16,
    )

    return output


def replacement_func():
    return conv1x1_bilinear_upsample_512


# ─── GEMM-only kernel (fp16/bf16 output) ─────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_fp16_out_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = acc + tl.dot(a, b, out_dtype=tl.float32)
    bias = tl.load(Bias_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    acc = acc + bias[:, None]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if IS_BFLOAT16:
        tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16), mask=c_mask)


@torch.fx.wrap
def conv1x1_fp16_out(in_10, in_8, in_7):
    """
    Triton 1x1 conv returning [1, C_out, H_in, W_in] in fp16/bf16.
    The subsequent F.interpolate in the graph upsamples normally (no double upsample).
    """
    device = in_10.device
    dtype  = in_10.dtype
    C_in  = in_10.shape[1]
    H_in  = in_10.shape[2]
    W_in  = in_10.shape[3]
    C_out = in_8.shape[0]
    HW    = H_in * W_in

    output  = torch.empty((1, C_out, H_in, W_in), device=device, dtype=dtype)
    is_bf16 = (dtype == torch.bfloat16)

    grid_g = lambda m: (triton.cdiv(C_out, m['BLOCK_M']), triton.cdiv(HW, m['BLOCK_N']))
    gemm_bias_fp16_out_kernel[grid_g](
        in_8, in_10, in_7, output,
        C_out, HW, C_in,
        IS_BFLOAT16=is_bf16,
    )
    return output


def replacement_func():
    return conv1x1_fp16_out