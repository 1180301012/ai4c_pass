import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Pattern: match BOTH linear AND batch_norm (the entire forward graph)
# Argument order mirrors the model exactly:
#   in_0=running_mean, in_1=running_var, in_2=bn_bias, in_3=bn_weight,
#   in_4=lin_bias,     in_5=lin_weight,  in_6=lin_x,   in_7=bn_x
# -----------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (linear, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # (lin_x, lin_w, lin_b,  bn_x, mean,  var,  bn_w,  bn_b)
    return (in_6, in_5, in_4, in_7, in_0, in_1, in_3, in_2)


# -----------------------------------------------------------------------
# Triton BN kernel  (no autotune – fixed BLOCK_C=512 for C=384)
# -----------------------------------------------------------------------

@triton.jit
def _bn_kernel(
    x_ptr, mean_ptr, var_ptr, w_ptr, b_ptr, out_ptr,
    B, C,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    mean   = tl.load(mean_ptr  + offs, mask=mask, other=0.0).to(tl.float32)
    var    = tl.load(var_ptr   + offs, mask=mask, other=1.0).to(tl.float32)
    w      = tl.load(w_ptr     + offs, mask=mask, other=1.0).to(tl.float32)
    b_coef = tl.load(b_ptr     + offs, mask=mask, other=0.0).to(tl.float32)

    scale = w / tl.sqrt(var + 1e-5)
    shift = b_coef - mean * scale

    x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0)
    out_f32 = x.to(tl.float32) * scale + shift
    tl.store(out_ptr + row * C + offs, out_f32.to(x.dtype), mask=mask)


# -----------------------------------------------------------------------
# Triton GEMM kernel  (no autotune – fixed tiles for K=384, N=1000)
# -----------------------------------------------------------------------

_DTYPE_MAP = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


@triton.jit
def _gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    sx0, sx1,
    sw0, sw1,
    so0, so1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    gsize = GROUP_M * num_n
    gid   = pid // gsize
    fm    = gid * GROUP_M
    gs    = min(num_m - fm, GROUP_M)
    pm    = fm + (pid % gsize) % gs
    pn    = (pid % gsize) // gs

    m_offs = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    m_mask = m_offs < M
    n_mask = n_offs < N

    xp = x_ptr + m_offs[:, None] * sx0 + k_offs[None, :] * sx1
    wp = w_ptr + n_offs[:, None] * sw0 + k_offs[None, :] * sw1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        km = (k * BLOCK_K + k_offs) < K
        xv = tl.load(xp, mask=m_mask[:, None] & km[None, :], other=0.0)
        wv = tl.load(wp, mask=n_mask[:, None] & km[None, :], other=0.0)
        acc += tl.dot(xv, tl.trans(wv), allow_tf32=True)
        xp += BLOCK_K * sx1
        wp += BLOCK_K * sw1

    bv = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    acc += bv[None, :]

    op = out_ptr + m_offs[:, None] * so0 + n_offs[None, :] * so1
    tl.store(op, acc.to(OUT_DTYPE), mask=m_mask[:, None] & n_mask[None, :])


# -----------------------------------------------------------------------
# Single fused wrapper  – ONE Python call for both ops
# -----------------------------------------------------------------------

@torch.fx.wrap
def fused_linear_and_bn(lin_x, lin_w, lin_b, bn_x, mean, var, bn_w, bn_b):
    M, K  = lin_x.shape[0], lin_x.shape[1]
    N     = lin_w.shape[0]
    B_bn  = bn_x.shape[0]
    C_bn  = bn_x.shape[1]

    lin_out = torch.empty((M, N), dtype=lin_x.dtype, device=lin_x.device)
    bn_out  = torch.empty_like(bn_x)

    OUT_DTYPE = _DTYPE_MAP[lin_x.dtype]

    # -- GEMM: fixed tile (no autotune) -----------------------------------
    BM, BN, BK, GM = 16, 64, 64, 8
    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN
    _gemm_kernel[(grid_m * grid_n,)](
        lin_x, lin_w, lin_b, lin_out,
        M, N, K,
        lin_x.stride(0), lin_x.stride(1),
        lin_w.stride(0), lin_w.stride(1),
        lin_out.stride(0), lin_out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
        OUT_DTYPE=OUT_DTYPE,
        num_warps=4, num_stages=3,
    )

    # -- BN: one program per row ------------------------------------------
    _bn_kernel[(B_bn,)](
        bn_x, mean, var, bn_w, bn_b, bn_out,
        B_bn, C_bn,
        BLOCK_C=512,
        num_warps=4, num_stages=2,
    )

    return (lin_out, bn_out)


def replacement_func():
    return fused_linear_and_bn