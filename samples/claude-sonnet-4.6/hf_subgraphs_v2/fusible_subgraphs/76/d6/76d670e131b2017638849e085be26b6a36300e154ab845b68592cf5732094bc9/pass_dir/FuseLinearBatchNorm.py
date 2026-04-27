import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (linear, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)


# ── Batch-norm inference kernel ───────────────────────────────────────────────
# Input x : [B, C]   running_mean/var : [C]   weight/bias : [C]
# out[b,c] = (x[b,c] - mean[c]) / sqrt(var[c]+eps) * weight[c] + bias[c]

@triton.jit
def bn_inference_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C,
    BLOCK_C: tl.constexpr,
    DTYPE: tl.constexpr,  # 0=float32, 1=float16, 2=bfloat16
):
    bid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_C)
    mask = offsets < C

    x       = tl.load(x_ptr       + bid * C + offsets, mask=mask, other=0.0)
    mean    = tl.load(mean_ptr     + offsets,            mask=mask, other=0.0)
    var     = tl.load(var_ptr      + offsets,            mask=mask, other=0.0)
    weight  = tl.load(weight_ptr   + offsets,            mask=mask, other=0.0)
    bias_v  = tl.load(bias_ptr     + offsets,            mask=mask, other=0.0)

    x_f32      = x.to(tl.float32)
    mean_f32   = mean.to(tl.float32)
    var_f32    = var.to(tl.float32)
    weight_f32 = weight.to(tl.float32)
    bias_f32   = bias_v.to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var_f32 + 1e-5)
    out_f32 = (x_f32 - mean_f32) * inv_std * weight_f32 + bias_f32

    if DTYPE == 1:
        tl.store(out_ptr + bid * C + offsets, out_f32.to(tl.float16), mask=mask)
    elif DTYPE == 2:
        tl.store(out_ptr + bid * C + offsets, out_f32.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + bid * C + offsets, out_f32, mask=mask)


# ── Linear GEMM kernel  ───────────────────────────────────────────────────────
# C[m,n] = sum_k x[m,k] * W[n,k] + bias[n]
#   x   : [M, K]   W   : [N, K]   bias : [N]   out : [M, N]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K', 'DTYPE'],
)
@triton.jit
def linear_gemm_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DTYPE:   tl.constexpr,  # 0=float32, 1=float16, 2=bfloat16
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    rk = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K
        k_rng = k_off + rk  # [BLOCK_K]

        # x tile  : [BLOCK_M, BLOCK_K]
        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + k_rng[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (k_rng[None, :] < K),
            other=0.0,
        )
        # w tile loaded as [BLOCK_K, BLOCK_N]  (W^T layout)
        w_tile = tl.load(
            w_ptr + rn[None, :] * stride_wn + k_rng[:, None] * stride_wk,
            mask=(rn[None, :] < N) & (k_rng[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(x_tile, w_tile)  # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]

    # add bias
    b = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc += b[None, :]

    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    out_offs = rm[:, None] * stride_om + rn[None, :] * stride_on

    # explicit cast to match output pointer dtype
    if DTYPE == 1:
        tl.store(out_ptr + out_offs, acc.to(tl.float16),  mask=out_mask)
    elif DTYPE == 2:
        tl.store(out_ptr + out_offs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptr + out_offs, acc,                  mask=out_mask)


# ── Replacement wrapper ───────────────────────────────────────────────────────

@torch.fx.wrap
def fused_linear_bn(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # in_0: running_mean [C]    in_1: running_var  [C]
    # in_2: bn_bias      [C]    in_3: bn_weight    [C]
    # in_4: linear_bias  [N]    in_5: linear_weight [N, K]
    # in_6: linear_input [B, K] in_7: bn_input     [B, C]

    B  = in_6.shape[0]
    K  = in_6.shape[1]
    N  = in_5.shape[0]
    C  = in_7.shape[1]

    # determine dtype flag once (Python-level branch, not restricted)
    dtype6 = in_6.dtype
    dtype7 = in_7.dtype
    if dtype6 == torch.float16:
        lin_dtype = 1
    elif dtype6 == torch.bfloat16:
        lin_dtype = 2
    else:
        lin_dtype = 0
    if dtype7 == torch.float16:
        bn_dtype = 1
    elif dtype7 == torch.bfloat16:
        bn_dtype = 2
    else:
        bn_dtype = 0

    # allocate outputs
    linear_out = torch.empty((B, N), dtype=dtype6, device=in_6.device)
    bn_out     = torch.empty_like(in_7)

    # ── Batch-norm inference ──────────────────────────────────────────────────
    BLOCK_C = 512  # next power-of-2 ≥ 384
    bn_inference_kernel[(B,)](
        in_7, in_0, in_1, in_3, in_2, bn_out,
        B, C,
        BLOCK_C=BLOCK_C,
        DTYPE=bn_dtype,
    )

    # ── Linear GEMM ───────────────────────────────────────────────────────────
    grid = lambda meta: (
        triton.cdiv(B, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_gemm_kernel[grid](
        in_6, in_5, in_4, linear_out,
        B, N, K,
        in_6.stride(0), in_6.stride(1),
        in_5.stride(0), in_5.stride(1),
        linear_out.stride(0), linear_out.stride(1),
        DTYPE=lin_dtype,
    )

    return (linear_out, bn_out)


def replacement_func():
    return fused_linear_bn