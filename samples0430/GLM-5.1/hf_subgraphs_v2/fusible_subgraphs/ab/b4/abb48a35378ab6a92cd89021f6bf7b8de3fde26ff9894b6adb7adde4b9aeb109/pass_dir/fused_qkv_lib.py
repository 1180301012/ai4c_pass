import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_qkv_matmul_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    weight_offset,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_oh, stride_od1, stride_od2,
    HEAD_DIM: tl.constexpr,
    IS_TRANSPOSED: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused QKV matmul kernel that writes directly to output with custom strides.
    
    Computes: output = input @ weight[weight_offset:weight_offset+N].T
    Writes result to output tensor with strides for (head, seq/dim, dim/seq) layout.
    
    For q/v (IS_TRANSPOSED=False): output[0, head, seq, dim] layout
    For k_t (IS_TRANSPOSED=True): output[0, head, dim, seq] layout
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Decompose offs_n into head index and dim-within-head index
    heads = offs_n // HEAD_DIM
    dims = offs_n % HEAD_DIM
    
    # Initialize accumulator in fp32 (tl.dot always accumulates in fp32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        
        # Load input tile: shape (BLOCK_M, BLOCK_K)
        # input[seq, embed] = input_ptr + seq * stride_im + embed * stride_ik
        a_ptrs = input_ptr + offs_m[:, None] * stride_im + k_offs[None, :] * stride_ik
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load weight tile (transposed for matmul): shape (BLOCK_K, BLOCK_N)
        # We compute: output = input @ weight_slice.T
        # So we need weight_slice.T which is (K, N), loaded as (BLOCK_K, BLOCK_N)
        # weight[n, k] = weight_ptr + (weight_offset + n) * stride_wn + k * stride_wk
        # We load b[k, n] = weight[weight_offset + n, k]
        b_ptrs = weight_ptr + (weight_offset + offs_n[None, :]) * stride_wn + k_offs[:, None] * stride_wk
        b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate using matrix multiplication (uses Tensor Cores when available)
        if DTYPE_ID == 0:  # fp32 - use TF32 for Tensor Core acceleration
            acc += tl.dot(a, b, allow_tf32=True)
        else:  # fp16/bf16 - Tensor Cores used automatically
            acc += tl.dot(a, b)
    
    # Cast accumulator to output dtype
    if DTYPE_ID == 0:
        result = acc  # fp32 output
    elif DTYPE_ID == 1:
        result = acc.to(tl.float16)  # fp16 output
    else:
        result = acc.to(tl.bfloat16)  # bf16 output
    
    # Compute output pointer offsets based on tensor layout
    m_mask = offs_m < M
    n_mask = offs_n < N
    mask = m_mask[:, None] & n_mask[None, :]
    
    if IS_TRANSPOSED:
        # k_t layout: (1, N_heads, HEAD_DIM, seq_len)
        # offs_m -> seq positions, heads/dims -> head and dim indices
        out_ptrs = output_ptr + heads[None, :] * stride_oh + dims[None, :] * stride_od1 + offs_m[:, None] * stride_od2
    else:
        # q/v layout: (1, N_heads, seq_len, HEAD_DIM)
        # offs_m -> seq positions, heads/dims -> head and dim indices
        out_ptrs = output_ptr + heads[None, :] * stride_oh + offs_m[:, None] * stride_od1 + dims[None, :] * stride_od2
    
    tl.store(out_ptrs, result, mask=mask)


@torch.fx.wrap
def fused_qkv_dispatch(*args):
    """Dispatch wrapper for fused QKV projection.
    
    Args:
        input_tensor: Input tensor of shape (1, seq_len, embed_dim)
        weight_tensor: Weight tensor of shape (3*n_heads*head_dim, embed_dim)
        route: Route string (e.g., "nheads4", "nheads9", "nheads16")
    
    Returns:
        (q, k_t, v) tuple where:
        - q: shape (1, n_heads, seq_len, head_dim)
        - k_t: shape (1, n_heads, head_dim, seq_len) (transposed k)
        - v: shape (1, n_heads, seq_len, head_dim)
    """
    input_tensor = args[0]
    weight_tensor = args[1]
    
    # Ensure weight is on the same device as input
    if weight_tensor.device != input_tensor.device:
        weight_tensor = weight_tensor.to(device=input_tensor.device)
    
    # Ensure weight has the same dtype as input
    if weight_tensor.dtype != input_tensor.dtype:
        weight_tensor = weight_tensor.to(dtype=input_tensor.dtype)
    
    # Derive dimensions from tensor shapes
    seq_len = input_tensor.shape[1]   # 197
    embed_dim = input_tensor.shape[2]  # varies: 192, 432, 768
    total_out = weight_tensor.shape[0]  # 3 * n_heads * head_dim
    out_per_type = total_out // 3       # n_heads * head_dim
    head_dim = 48                        # always 48 for these models
    n_heads = out_per_type // head_dim   # varies: 4, 9, 16
    
    # Detect dtype for kernel specialization
    dtype = input_tensor.dtype
    if dtype == torch.float32:
        dtype_id = 0
    elif dtype == torch.float16:
        dtype_id = 1
    else:  # bfloat16
        dtype_id = 2
    
    # Create output tensors with correct shapes
    q = torch.empty(1, n_heads, seq_len, head_dim, dtype=dtype, device=input_tensor.device)
    k_t = torch.empty(1, n_heads, head_dim, seq_len, dtype=dtype, device=input_tensor.device)
    v = torch.empty(1, n_heads, seq_len, head_dim, dtype=dtype, device=input_tensor.device)
    
    # Get strides for input and weight
    stride_im = input_tensor.stride(1)  # stride for seq dimension
    stride_ik = input_tensor.stride(2)  # stride for embed dimension
    stride_wn = weight_tensor.stride(0)  # stride for output dim
    stride_wk = weight_tensor.stride(1)  # stride for embed dim
    
    # Matmul dimensions
    M = seq_len
    N = out_per_type
    K = embed_dim
    
    # Grid function for autotuning
    def grid_fn(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    # Launch kernel for q (not transposed, weight_offset=0)
    fused_qkv_matmul_kernel[grid_fn](
        input_tensor, weight_tensor, q,
        M, N, K,
        0,  # weight_offset for q
        stride_im, stride_ik, stride_wn, stride_wk,
        q.stride(1), q.stride(2), q.stride(3),
        HEAD_DIM=head_dim, IS_TRANSPOSED=False, DTYPE_ID=dtype_id,
    )
    
    # Launch kernel for k_t (transposed, weight_offset=N)
    fused_qkv_matmul_kernel[grid_fn](
        input_tensor, weight_tensor, k_t,
        M, N, K,
        N,  # weight_offset for k
        stride_im, stride_ik, stride_wn, stride_wk,
        k_t.stride(1), k_t.stride(2), k_t.stride(3),
        HEAD_DIM=head_dim, IS_TRANSPOSED=True, DTYPE_ID=dtype_id,
    )
    
    # Launch kernel for v (not transposed, weight_offset=2*N)
    fused_qkv_matmul_kernel[grid_fn](
        input_tensor, weight_tensor, v,
        M, N, K,
        2 * N,  # weight_offset for v
        stride_im, stride_ik, stride_wn, stride_wk,
        v.stride(1), v.stride(2), v.stride(3),
        HEAD_DIM=head_dim, IS_TRANSPOSED=False, DTYPE_ID=dtype_id,
    )
    
    return q, k_t, v