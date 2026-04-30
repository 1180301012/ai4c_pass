import torch
import triton
import triton.language as tl

# Constants for the EfficientFormer QKV projection
SEQ_LEN = 49
NUM_HEADS = 8
HEAD_DIM = 192  # Q_DIM + K_DIM + V_DIM = 32 + 32 + 128
Q_DIM = 32
K_DIM = 32
V_DIM = 128
TOTAL_DIM = NUM_HEADS * HEAD_DIM  # 1536


@triton.jit
def fused_qkv_matmul_kernel(
    input_ptr, weight_ptr, bias_ptr,
    q_ptr, kt_ptr, v_ptr,
    M, K, N,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_bn,
    stride_q0, stride_q1, stride_q2, stride_q3,
    stride_kt0, stride_kt1, stride_kt2, stride_kt3,
    stride_v0, stride_v1, stride_v2, stride_v3,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SEQ_LEN_C: tl.constexpr, HEAD_DIM_C: tl.constexpr,
    Q_DIM_C: tl.constexpr, K_DIM_C: tl.constexpr, V_DIM_C: tl.constexpr,
    NUM_HEADS_C: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # 2D grid: programs are organized as (pid_m, pid_n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main matmul loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k_cur = k_start + offs_k
        
        # Load input tile: (BLOCK_M, BLOCK_K) - contiguous along K
        a_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k_cur[None, :] * stride_ik
        a_mask = (offs_m[:, None] < M) & (offs_k_cur[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load weight tile: (BLOCK_N, BLOCK_K) - contiguous along K, then transpose for dot
        # weight shape is (N, K) with strides (stride_wn, stride_wk)
        # We load weight[n, k] for n in offs_n, k in offs_k_cur
        b_ptrs = weight_ptr + offs_n[:, None] * stride_wn + offs_k_cur[None, :] * stride_wk
        b_mask = (offs_n[:, None] < N) & (offs_k_cur[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # b is (BLOCK_N, BLOCK_K), transpose to (BLOCK_K, BLOCK_N) for dot
        acc += tl.dot(a, tl.trans(b), allow_tf32=True)
    
    # Add bias
    bias_ptrs = bias_ptr + offs_n * stride_bn
    bias_mask = offs_n < N
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc += bias[None, :]
    
    # Output mask
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Route output to Q, K_t, or V based on n_start
    head_idx = n_start // HEAD_DIM_C
    lc_start = n_start % HEAD_DIM_C
    
    # Compute batch and sequence indices from m offsets
    b_idx = offs_m // SEQ_LEN_C
    s_idx = offs_m % SEQ_LEN_C
    
    # Compute local channel offsets within the head
    lc_offsets = offs_n - head_idx * HEAD_DIM_C
    
    # Cast accumulator to output dtype
    out_vals = acc.to(OUT_DTYPE)
    
    if lc_start < Q_DIM_C:
        # Q output: shape (batch, num_heads, seq_len, q_dim)
        # Q[b, h, s, lc] where lc = lc_offsets
        q_ptrs = q_ptr + b_idx[:, None] * stride_q0 + head_idx * stride_q1 + s_idx[:, None] * stride_q2 + lc_offsets[None, :] * stride_q3
        tl.store(q_ptrs, out_vals, mask=out_mask)
    elif lc_start < Q_DIM_C + K_DIM_C:
        # K_t output: shape (batch, num_heads, k_dim, seq_len)
        # K_t[b, h, dk, s] where dk = lc_offsets - Q_DIM_C
        dk_offsets = lc_offsets - Q_DIM_C
        kt_ptrs = kt_ptr + b_idx[:, None] * stride_kt0 + head_idx * stride_kt1 + dk_offsets[None, :] * stride_kt2 + s_idx[:, None] * stride_kt3
        tl.store(kt_ptrs, out_vals, mask=out_mask)
    else:
        # V output: shape (batch, num_heads, seq_len, v_dim)
        # V[b, h, s, dv] where dv = lc_offsets - Q_DIM_C - K_DIM_C
        dv_offsets = lc_offsets - Q_DIM_C - K_DIM_C
        v_ptrs = v_ptr + b_idx[:, None] * stride_v0 + head_idx * stride_v1 + s_idx[:, None] * stride_v2 + dv_offsets[None, :] * stride_v3
        tl.store(v_ptrs, out_vals, mask=out_mask)


@torch.fx.wrap
def fused_qkv_dispatch(in_1, in_2, in_3, route):
    """Dispatch wrapper for fused QKV projection kernel.
    
    This function is shared by all pass files via routing.
    The route parameter identifies which pass matched, but the computation
    is the same for all routes (batch size is determined from input shape).
    
    Args:
        in_1: bias tensor (1536,) on CUDA
        in_2: weight tensor (1536, 448) on CUDA  
        in_3: input tensor (batch, 49, 448) on CUDA
        route: routing string (ignored, all routes do same computation)
    
    Returns:
        (Q, K_t, V) where:
        - Q: (batch, 8, 49, 32) 
        - K_t: (batch, 8, 32, 49)
        - V: (batch, 8, 49, 128)
    """
    batch = in_3.shape[0]
    M = batch * SEQ_LEN
    K = in_3.shape[-1]  # 448
    N = TOTAL_DIM  # 1536
    
    dtype = in_3.dtype
    device = in_3.device
    
    # Allocate output tensors
    Q = torch.empty((batch, NUM_HEADS, SEQ_LEN, Q_DIM), dtype=dtype, device=device)
    K_t = torch.empty((batch, NUM_HEADS, K_DIM, SEQ_LEN), dtype=dtype, device=device)
    V = torch.empty((batch, NUM_HEADS, SEQ_LEN, V_DIM), dtype=dtype, device=device)
    
    # Flatten input for 2D matmul
    input_2d = in_3.reshape(-1, K)
    
    # Determine output dtype for Triton constexpr
    if dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif dtype == torch.float16:
        out_dtype = tl.float16
    else:
        out_dtype = tl.float32
    
    # Determine block sizes based on M for better utilization
    if M <= 64:
        BLOCK_M = 16
        BLOCK_N = 32
        BLOCK_K = 32
    elif M <= 256:
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32
    elif M <= 1024:
        BLOCK_M = 64
        BLOCK_N = 32
        BLOCK_K = 32
    else:
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
    
    # Launch kernel with 2D grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    fused_qkv_matmul_kernel[grid](
        input_2d, in_2, in_1,
        Q, K_t, V,
        M, K, N,
        input_2d.stride(0), input_2d.stride(1),
        in_2.stride(0), in_2.stride(1),
        in_1.stride(0),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_t.stride(0), K_t.stride(1), K_t.stride(2), K_t.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SEQ_LEN_C=SEQ_LEN, HEAD_DIM_C=HEAD_DIM,
        Q_DIM_C=Q_DIM, K_DIM_C=K_DIM, V_DIM_C=V_DIM,
        NUM_HEADS_C=NUM_HEADS,
        OUT_DTYPE=out_dtype,
    )
    
    return (Q, K_t, V)