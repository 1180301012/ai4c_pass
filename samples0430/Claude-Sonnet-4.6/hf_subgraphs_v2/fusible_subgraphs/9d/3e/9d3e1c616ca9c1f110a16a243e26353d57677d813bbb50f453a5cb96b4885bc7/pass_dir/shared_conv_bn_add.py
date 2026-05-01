import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_bn_add_kernel(
    # Pointers
    input_ptr,
    weight_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    residual_ptr,
    output_ptr,
    # Dimensions (runtime)
    M, K, N, HW,
    # BN epsilon (compile-time constant)
    BN_EPS: tl.constexpr,
    # dtype flags (compile-time)
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    # Tile sizes (compile-time)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused 1×1 Conv + BN (inference) + residual add.

    Input layout : NCHW → treated as [M=N*H*W, K=C_in]  (stride-K along spatial axis)
    Weight layout: [N=C_out, K=C_in, 1, 1]               (contiguous [N, K])
    Output layout: NCHW → [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)  # [BM]
    n_range = n_start + tl.arange(0, BLOCK_N)  # [BN]

    # For NCHW: pixel p = n*HW + hw
    batch_idx = m_range // HW   # [BM]
    hw_idx    = m_range % HW    # [BM]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------------------------------------------------ #
    # Accumulate over input channels (K dimension)                         #
    # ------------------------------------------------------------------ #
    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)  # [BK]

        # A tile  [BM, BK]
        # input[batch, k, hw]  flat = batch*(K*HW) + k*HW + hw
        A_idx  = batch_idx[:, None] * (K * HW) + k_range[None, :] * HW + hw_idx[:, None]
        A_mask = (m_range[:, None] < M) & (k_range[None, :] < K)
        A = tl.load(input_ptr + A_idx, mask=A_mask, other=0.0).to(tl.float32)

        # B tile  [BK, BN]
        # weight[oc, ic, 0, 0]  flat = oc*K + ic
        B_idx  = k_range[:, None] * N + n_range[None, :]   # Note: weight shape is [N, K], so flat = n*K + k → we need [k, n]
        # weight is stored as [C_out, C_in]: weight[n, k] → flat = n*K + k
        B_idx  = n_range[None, :] * K + k_range[:, None]   # [BK, BN]: B[k, n] = weight[n, k]
        B_mask = (k_range[:, None] < K) & (n_range[None, :] < N)
        B = tl.load(weight_ptr + B_idx, mask=B_mask, other=0.0).to(tl.float32)

        acc = tl.dot(A, B, acc)

    # ------------------------------------------------------------------ #
    # Fused BN scale+offset epilogue                                       #
    # ------------------------------------------------------------------ #
    n_mask  = n_range < N
    bn_mean = tl.load(bn_mean_ptr   + n_range, mask=n_mask, other=0.0).to(tl.float32)
    bn_var  = tl.load(bn_var_ptr    + n_range, mask=n_mask, other=1.0).to(tl.float32)
    bn_w    = tl.load(bn_weight_ptr + n_range, mask=n_mask, other=1.0).to(tl.float32)
    bn_b    = tl.load(bn_bias_ptr   + n_range, mask=n_mask, other=0.0).to(tl.float32)

    scale  = bn_w / tl.sqrt(bn_var + BN_EPS)          # [BN]
    offset = bn_b - bn_mean * scale                     # [BN]

    acc = acc * scale[None, :] + offset[None, :]        # [BM, BN]

    # ------------------------------------------------------------------ #
    # Add residual and store                                               #
    # ------------------------------------------------------------------ #
    # residual[batch, n, hw]  flat = batch*(N*HW) + n*HW + hw
    out_idx  = batch_idx[:, None] * (N * HW) + n_range[None, :] * HW + hw_idx[:, None]
    out_mask = (m_range[:, None] < M) & (n_range[None, :] < N)

    res = tl.load(residual_ptr + out_idx, mask=out_mask, other=0.0).to(tl.float32)
    acc = acc + res

    if IS_FP16:
        tl.store(output_ptr + out_idx, acc.to(tl.float16),   mask=out_mask)
    elif IS_BF16:
        tl.store(output_ptr + out_idx, acc.to(tl.bfloat16),  mask=out_mask)
    else:
        tl.store(output_ptr + out_idx, acc,                   mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_bn_add(x, weight, running_mean, running_var, bn_weight, bn_bias, residual):
    """
    Fused 1×1 Conv2d + BatchNorm (inference) + residual add.

    Args:
        x        : [N, C_in,  H, W]  – conv input
        weight   : [C_out, C_in, 1, 1] – conv weight
        running_mean, running_var : [C_out]
        bn_weight, bn_bias        : [C_out]
        residual : [N, C_out, H, W]  – skip-connection input
    Returns:
        output   : [N, C_out, H, W]
    """
    N_batch = x.shape[0]
    C_in    = x.shape[1]
    H       = x.shape[2]
    W       = x.shape[3]
    C_out   = weight.shape[0]
    HW      = H * W
    M       = N_batch * HW

    output = torch.empty(N_batch, C_out, H, W, dtype=x.dtype, device=x.device)

    is_fp16 = (x.dtype == torch.float16)
    is_bf16 = (x.dtype == torch.bfloat16)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(C_out, meta['BLOCK_N']))

    _conv1x1_bn_add_kernel[grid](
        x, weight,
        running_mean, running_var, bn_weight, bn_bias,
        residual, output,
        M, C_in, C_out, HW,
        1e-5,          # BN_EPS
        is_fp16,       # IS_FP16
        is_bf16,       # IS_BF16
    )

    return output