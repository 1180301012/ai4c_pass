import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    hs = torch.nn.functional.hardsigmoid(conv, False)
    mul = in_2 * hs
    pool = torch.nn.functional.adaptive_avg_pool2d(mul, 1)
    flat = pool.flatten(1, -1)
    out = torch.nn.functional.dropout(flat, 0.0, False, False)
    return (out,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ─── Kernel 1: GEMM (1x1 conv as matmul) + bias + hardsigmoid ─────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 4},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_hardsigmoid_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Swizzled program IDs for better L2 cache utilization
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k  = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Accumulate in float32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # hardsigmoid: clamp(x/6 + 0.5, 0, 1)
    acc = acc * (1.0 / 6.0) + 0.5
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)

    # Store result
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_c)


# ─── Kernel 2: Fused scale × avg_pool (avoids materialising [B,C,H,W]) ────────
@triton.jit
def scale_avg_pool_kernel(
    feat_ptr, scale_ptr, out_ptr,
    B, C, HW,
    stride_feat_b, stride_feat_c,
    stride_scale_b, stride_scale_c,
    stride_out_b,  stride_out_c,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Scalar scale for this (b, c)
    scale = tl.load(scale_ptr + b * stride_scale_b + c * stride_scale_c).to(tl.float32)

    # Load contiguous HW elements of in_2[b, c, :, :]
    feat_base = feat_ptr + b * stride_feat_b + c * stride_feat_c
    hw_offs   = tl.arange(0, BLOCK_HW)
    feat_vals = tl.load(feat_base + hw_offs, mask=hw_offs < HW, other=0.0).to(tl.float32)

    # Mean × scale
    feat_sum = tl.sum(feat_vals, axis=0)
    result   = scale * feat_sum / HW

    tl.store(out_ptr + b * stride_out_b + c * stride_out_c,
             result.to(out_ptr.dtype.element_ty))


# ─── Host wrapper ──────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_conv_hs_mul_pool(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [C]
    in_1 : weight [C, C, 1, 1]
    in_2 : x_feat [B, C, H, W]
    in_3 : x_se   [B, C, 1, 1]
    returns (output,)  shape [B, C]
    """
    B  = in_3.shape[0]
    C  = in_3.shape[1]
    H  = in_2.shape[2]
    W  = in_2.shape[3]
    HW = H * W

    # ── Kernel 1: GEMM + bias + hardsigmoid ──────────────────────────────────
    # 1×1 conv  ≡  A @ weight.T + bias
    # A = x_se viewed as [B, C];  weight viewed as [C, C]
    scale = torch.empty((B, C), dtype=in_3.dtype, device=in_3.device)
    output = torch.empty((B, C), dtype=in_2.dtype, device=in_2.device)

    M, N, K = B, C, C
    grid_gemm = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    # in_3 [B, C, 1, 1]: stride(0)=C, stride(1)=1
    # in_1 [C, C, 1, 1] used as B^T: stride_bk=in_1.stride(1)=1, stride_bn=in_1.stride(0)=C
    gemm_bias_hardsigmoid_kernel[grid_gemm](
        in_3, in_1, in_0, scale,
        M, N, K,
        in_3.stride(0), in_3.stride(1),   # A strides
        in_1.stride(1), in_1.stride(0),   # B strides (transposed)
        scale.stride(0), scale.stride(1), # C strides
    )

    # ── Kernel 2: scale × avg_pool ────────────────────────────────────────────
    # Choose BLOCK_HW as next power-of-2 >= HW, capped at 512
    if HW <= 64:
        BLOCK_HW = 64
    elif HW <= 128:
        BLOCK_HW = 128
    elif HW <= 256:
        BLOCK_HW = 256
    else:
        BLOCK_HW = 512

    scale_avg_pool_kernel[(B * C,)](
        in_2, scale, output,
        B, C, HW,
        in_2.stride(0), in_2.stride(1),     # feat strides (b, c)
        scale.stride(0), scale.stride(1),   # scale strides
        output.stride(0), output.stride(1), # out strides
        BLOCK_HW=BLOCK_HW,
    )

    return (output,)


def replacement_func():
    return fused_conv_hs_mul_pool