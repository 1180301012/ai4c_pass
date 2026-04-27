import torch
import triton
import triton.language as tl


# ── Batch-norm inference kernel ───────────────────────────────────────────────
@triton.jit
def bn_inference_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C,
    BLOCK_C: tl.constexpr,
    DTYPE: tl.constexpr,   # 0=fp32  1=fp16  2=bf16
):
    bid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    x      = tl.load(x_ptr      + bid * C + offs, mask=mask, other=0.0)
    mean   = tl.load(mean_ptr   + offs,            mask=mask, other=0.0)
    var    = tl.load(var_ptr    + offs,            mask=mask, other=0.0)
    wt     = tl.load(weight_ptr + offs,            mask=mask, other=0.0)
    bias_v = tl.load(bias_ptr   + offs,            mask=mask, other=0.0)

    xf  = x.to(tl.float32)
    mf  = mean.to(tl.float32)
    vf  = var.to(tl.float32)
    wf  = wt.to(tl.float32)
    bf  = bias_v.to(tl.float32)

    out_f32 = (xf - mf) * (1.0 / tl.sqrt(vf + 1e-5)) * wf + bf

    if DTYPE == 1:
        tl.store(out_ptr + bid * C + offs, out_f32.to(tl.float16),  mask=mask)
    elif DTYPE == 2:
        tl.store(out_ptr + bid * C + offs, out_f32.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + bid * C + offs, out_f32,                  mask=mask)


# ── Linear GEMM kernel  ───────────────────────────────────────────────────────
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
    DTYPE:   tl.constexpr,   # 0=fp32  1=fp16  2=bf16
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K
        k_rng = k_off + rk

        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + k_rng[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (k_rng[None, :] < K), other=0.0,
        )
        w_tile = tl.load(                               # [BLOCK_N, BLOCK_K] coalesced
            w_ptr + rn[:, None] * stride_wn + k_rng[None, :] * stride_wk,
            mask=(rn[:, None] < N) & (k_rng[None, :] < K), other=0.0,
        )
        acc += tl.dot(x_tile, tl.trans(w_tile))  # [BLOCK_M,BLOCK_K] @ [BLOCK_K,BLOCK_N]

    b = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc += b[None, :]

    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    out_offs = rm[:, None] * stride_om + rn[None, :] * stride_on

    if DTYPE == 1:
        tl.store(out_ptr + out_offs, acc.to(tl.float16),  mask=out_mask)
    elif DTYPE == 2:
        tl.store(out_ptr + out_offs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptr + out_offs, acc,                  mask=out_mask)


# ── Shared dispatch wrapper (one replacement_func for both passes) ─────────────

@torch.fx.wrap
def shared_dispatch(*args):
    """
    Route to the correct Triton kernel based on the last string argument.
    BN route:     args = (x, mean, var, weight, bias,  "bn")
    Linear route: args = (x, weight, bias,              "linear")
    """
    route = args[-1]

    if route == "bn":
        in_7, in_0, in_1, in_3, in_2 = args[0], args[1], args[2], args[3], args[4]
        B  = in_7.shape[0]
        C  = in_7.shape[1]
        dt = in_7.dtype
        if dt == torch.float16:
            bn_dtype = 1
        elif dt == torch.bfloat16:
            bn_dtype = 2
        else:
            bn_dtype = 0
        bn_out = torch.empty((B, C), dtype=dt, device=in_7.device)
        bn_inference_kernel[(B,)](
            in_7, in_0, in_1, in_3, in_2, bn_out,
            B, C,
            BLOCK_C=512,
            DTYPE=bn_dtype,
        )
        return bn_out

    elif route == "linear":
        in_6, in_5, in_4 = args[0], args[1], args[2]
        B  = in_6.shape[0]
        K  = in_6.shape[1]
        N  = in_5.shape[0]
        dt = in_6.dtype
        if dt == torch.float16:
            lin_dtype = 1
        elif dt == torch.bfloat16:
            lin_dtype = 2
        else:
            lin_dtype = 0
        linear_out = torch.empty((B, N), dtype=dt, device=in_6.device)
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_M']),
                             triton.cdiv(N, meta['BLOCK_N']))
        # Assume contiguous row-major layout (true for all standard weight tensors)
        # This avoids calling .stride() on PoisonDispatchTensors (dispatch overhead)
        linear_gemm_kernel[grid](
            in_6, in_5, in_4, linear_out,
            B, N, K,
            K, 1,   # x   strides: (stride_m=K, stride_k=1)
            K, 1,   # W   strides: (stride_n=K, stride_k=1)
            N, 1,   # out strides: (stride_m=N, stride_k=1)
            DTYPE=lin_dtype,
        )
        return linear_out