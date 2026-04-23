import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n * stride_bias, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]

    c = accumulator.to(c_ptr.dtype.element_ty)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def rearrange_q_kernel(
    linear_ptr, q_ptr,
    B, S, N_HEADS, HEAD_DIM, Q_DIM,
    stride_lb, stride_ls, stride_lo,
    stride_qb, stride_qh, stride_qs, stride_qd,
    BLOCK_B: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """Read Q values from linear output and write to Q tensor."""
    pid = tl.program_id(axis=0)
    num_b = tl.cdiv(B, BLOCK_B)
    num_s = tl.cdiv(S, BLOCK_S)
    pid_h = pid // (num_b * num_s)
    pid_bs = pid % (num_b * num_s)
    pid_b = pid_bs // num_s
    pid_s = pid_bs % num_s

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_dq = tl.arange(0, Q_DIM)
    h = pid_h

    b_mask = offs_b < B
    s_mask = offs_s < S

    # linear[b, s, h*HEAD_DIM + dq] for dq in 0..Q_DIM-1
    o_idx = h * HEAD_DIM + offs_dq
    linear_ptrs = linear_ptr + offs_b[:, None] * stride_lb + offs_s[None, :] * stride_ls + o_idx[None, :] * stride_lo
    mask = b_mask[:, None] & s_mask[None, :]
    vals = tl.load(linear_ptrs, mask=mask, other=0.0)

    # Q[b, h, s, dq]
    q_ptrs = q_ptr + offs_b[:, None] * stride_qb + h * stride_qh + offs_s[None, :] * stride_qs + offs_dq[None, :] * stride_qd
    tl.store(q_ptrs, vals, mask=mask)


@triton.jit
def rearrange_k_t_kernel(
    linear_ptr, k_t_ptr,
    B, S, N_HEADS, HEAD_DIM, Q_DIM, K_DIM,
    stride_lb, stride_ls, stride_lo,
    stride_kb, stride_kh, stride_kd, stride_ks,
    BLOCK_B: tl.constexpr, BLOCK_DK: tl.constexpr,
):
    """Read K values from linear output and write transposed to K_t tensor."""
    pid = tl.program_id(axis=0)
    num_b = tl.cdiv(B, BLOCK_B)
    num_dk = tl.cdiv(K_DIM, BLOCK_DK)
    pid_h = pid // (num_b * num_dk)
    pid_bdk = pid % (num_b * num_dk)
    pid_b = pid_bdk // num_dk
    pid_dk = pid_bdk % num_dk

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_dk = pid_dk * BLOCK_DK + tl.arange(0, BLOCK_DK)
    offs_s = tl.arange(0, S)
    h = pid_h

    b_mask = offs_b < B
    dk_mask = offs_dk < K_DIM
    s_mask = offs_s < S

    # linear[b, s, h*HEAD_DIM + Q_DIM + dk] for s in 0..S-1, dk in 0..K_DIM-1
    o_idx = h * HEAD_DIM + Q_DIM + offs_dk
    # We need to load: for each (b, dk), load S values along the s dimension
    # linear[b, s, o_idx] - stride_ls for varying s
    linear_ptrs = linear_ptr + offs_b[:, None, None] * stride_lb + offs_s[None, :, None] * stride_ls + o_idx[None, None, :] * stride_lo
    mask = b_mask[:, None, None] & s_mask[None, :, None] & dk_mask[None, None, :]
    vals = tl.load(linear_ptrs, mask=mask, other=0.0)

    # K_t[b, h, dk, s] - note the transpose: dk varies along dim 2, s varies along dim 3
    k_ptrs = k_t_ptr + offs_b[:, None, None] * stride_kb + h * stride_kh + offs_dk[None, None, :] * stride_kd + offs_s[None, :, None] * stride_ks
    tl.store(k_ptrs, vals, mask=mask)


@triton.jit
def rearrange_v_kernel(
    linear_ptr, v_ptr,
    B, S, N_HEADS, HEAD_DIM, Q_DIM, K_DIM, V_DIM,
    stride_lb, stride_ls, stride_lo,
    stride_vb, stride_vh, stride_vs, stride_vd,
    BLOCK_B: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """Read V values from linear output and write to V tensor."""
    pid = tl.program_id(axis=0)
    num_b = tl.cdiv(B, BLOCK_B)
    num_s = tl.cdiv(S, BLOCK_S)
    pid_h = pid // (num_b * num_s)
    pid_bs = pid % (num_b * num_s)
    pid_b = pid_bs // num_s
    pid_s = pid_bs % num_s

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_dv = tl.arange(0, V_DIM)
    h = pid_h

    b_mask = offs_b < B
    s_mask = offs_s < S

    # linear[b, s, h*HEAD_DIM + Q_DIM + K_DIM + dv] for dv in 0..V_DIM-1
    o_idx = h * HEAD_DIM + Q_DIM + K_DIM + offs_dv
    linear_ptrs = linear_ptr + offs_b[:, None] * stride_lb + offs_s[None, :] * stride_ls + o_idx[None, :] * stride_lo
    mask = b_mask[:, None] & s_mask[None, :]
    vals = tl.load(linear_ptrs, mask=mask, other=0.0)

    # V[b, h, s, dv]
    v_ptrs = v_ptr + offs_b[:, None] * stride_vb + h * stride_vh + offs_s[None, :] * stride_vs + offs_dv[None, :] * stride_vd
    tl.store(v_ptrs, vals, mask=mask)


@torch.fx.wrap
def fused_qkv_wrapper(in_0, in_1, in_2, in_3):
    """
    Fused QKV projection + rearrangement kernel.
    
    Computes: linear = in_3 @ in_2.T + in_1  (shape [B, 49, 1536])
    Then rearranges into:
      Q: [B, 8, 49, 32]
      K_t: [B, 8, 32, 49]  (K transposed)
      V: [B, 8, 49, 128]
    Plus transfers in_0 from CPU to CUDA.
    """
    # Constants for this specific model
    S = 49
    N_HEADS = 8
    Q_DIM = 32
    K_DIM = 32
    V_DIM = 128
    HEAD_DIM = Q_DIM + K_DIM + V_DIM  # 192
    N_OUT = N_HEADS * HEAD_DIM  # 1536
    K_IN = 448
    
    B = in_3.shape[0]
    M = B * S
    
    # Step 1: Matmul + bias → linear output [B, 49, 1536]
    linear_out = torch.empty((B, S, N_OUT), dtype=in_3.dtype, device=in_3.device)
    
    stride_am = in_3.stride()[1]
    stride_ak = in_3.stride()[2]
    stride_bn = in_2.stride()[0]
    stride_bk = in_2.stride()[1]
    stride_cm = linear_out.stride()[1]
    stride_cn = linear_out.stride()[2]
    stride_bias = in_1.stride()[0]
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N_OUT, META['BLOCK_N']),)
    
    matmul_kernel[grid](
        in_3, in_2, linear_out, in_1,
        M, N_OUT, K_IN,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_bias,
    )
    
    # Step 2: Rearrange linear output into Q, K_t, V
    q_out = torch.empty((B, N_HEADS, S, Q_DIM), dtype=in_3.dtype, device=in_3.device)
    k_t_out = torch.empty((B, N_HEADS, K_DIM, S), dtype=in_3.dtype, device=in_3.device)
    v_out = torch.empty((B, N_HEADS, S, V_DIM), dtype=in_3.dtype, device=in_3.device)
    
    # Q rearrangement: each program handles one (b_chunk, s_chunk, h) triplet
    BLOCK_B_Q = 1
    BLOCK_S_Q = 1
    grid_q = (N_HEADS * triton.cdiv(B, BLOCK_B_Q) * triton.cdiv(S, BLOCK_S_Q),)
    
    rearrange_q_kernel[grid_q](
        linear_out, q_out,
        B, S, N_HEADS, HEAD_DIM, Q_DIM,
        linear_out.stride()[0], linear_out.stride()[1], linear_out.stride()[2],
        q_out.stride()[0], q_out.stride()[1], q_out.stride()[2], q_out.stride()[3],
        BLOCK_B=BLOCK_B_Q, BLOCK_S=BLOCK_S_Q,
    )
    
    # K_t rearrangement: each program handles one (b_chunk, dk_chunk, h) triplet
    BLOCK_B_K = 1
    BLOCK_DK_K = K_DIM  # all dk values at once
    grid_k = (N_HEADS * triton.cdiv(B, BLOCK_B_K) * triton.cdiv(K_DIM, BLOCK_DK_K),)
    
    rearrange_k_t_kernel[grid_k](
        linear_out, k_t_out,
        B, S, N_HEADS, HEAD_DIM, Q_DIM, K_DIM,
        linear_out.stride()[0], linear_out.stride()[1], linear_out.stride()[2],
        k_t_out.stride()[0], k_t_out.stride()[1], k_t_out.stride()[2], k_t_out.stride()[3],
        BLOCK_B=BLOCK_B_K, BLOCK_DK=BLOCK_DK_K,
    )
    
    # V rearrangement: each program handles one (b_chunk, s_chunk, h) triplet
    BLOCK_B_V = 1
    BLOCK_S_V = 1
    grid_v = (N_HEADS * triton.cdiv(B, BLOCK_B_V) * triton.cdiv(S, BLOCK_S_V),)
    
    rearrange_v_kernel[grid_v](
        linear_out, v_out,
        B, S, N_HEADS, HEAD_DIM, Q_DIM, K_DIM, V_DIM,
        linear_out.stride()[0], linear_out.stride()[1], linear_out.stride()[2],
        v_out.stride()[0], v_out.stride()[1], v_out.stride()[2], v_out.stride()[3],
        BLOCK_B=BLOCK_B_V, BLOCK_S=BLOCK_S_V,
    )
    
    # Step 3: Transfer in_0 from CPU to CUDA
    in_0_cuda = torch.as_tensor(in_0, device=in_3.device)
    
    return (q_out, in_0_cuda, k_t_out, v_out)