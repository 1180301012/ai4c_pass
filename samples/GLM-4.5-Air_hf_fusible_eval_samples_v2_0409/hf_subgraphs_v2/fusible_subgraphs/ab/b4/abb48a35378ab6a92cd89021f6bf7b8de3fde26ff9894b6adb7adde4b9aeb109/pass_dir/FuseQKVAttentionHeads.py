import torch
import triton
import triton.language as tl

def pattern(weight, x):
    # Linear operation: compute QKV
    linear = torch.nn.functional.linear(x, weight, None)
    
    # Reshape and permute for attention heads
    tmp_2 = linear.reshape(1, 197, 3, -1, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    
    # Split into Q, K, V
    tmp_5 = tmp_3[0]
    tmp_6 = tmp_3[1]
    tmp_7 = tmp_3[2]
    
    # Transpose K tensor
    tmp_8 = tmp_6.transpose(-2, -1)
    
    return tmp_5, tmp_8, tmp_7

def replacement_args(weight, x):
    return (weight, x)

@triton.jit
def fused_qkv_kernel(
    weight_ptr,
    x_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    num_heads,
    seq_len,
    head_dim,
    batch_size,
    weight_stride_0,
    weight_stride_1,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    q_stride_0,
    q_stride_1,
    q_stride_2,
    q_stride_3,
    k_stride_0,
    k_stride_1,
    k_stride_2,
    k_stride_3,
    v_stride_0,
    v_stride_1,
    v_stride_2,
    v_stride_3,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Compute program id
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Block ranges
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    k_start = pid_k * BLOCK_K
    
    # Create offsets
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    
    # Create masks
    m_mask = m_offsets < seq_len
    n_mask = n_offsets < (num_heads * head_dim)
    k_mask = k_offsets < x.size(2)
    
    # Load weight (3, num_heads, head_dim, in_features)
    weight_ptrs = weight_ptr + (pid_n // head_dim) * weight_stride_0 + (pid_n % head_dim) * weight_stride_1 + k_offsets[:, None] * weight_stride_2
    weight = tl.load(weight_ptrs, mask=k_mask[:, None], other=0.0)
    
    # Load input (batch_size, seq_len, in_features)
    x_ptrs = x_ptr + m_offsets[:, None] * x_stride_0 + k_offsets[None, :] * x_stride_2
    x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    
    # Compute matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, x.size(2), BLOCK_K):
        k_offsets_fixed = k + tl.arange(0, BLOCK_K)
        k_mask_fixed = k_offsets_fixed < x.size(2)
        
        # Load x
        x_ptrs = x_ptr + m_offsets[:, None] * x_stride_0 + k_offsets_fixed[None, :] * x_stride_2
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask_fixed[None, :], other=0.0)
        
        # Load weight
        weight_ptrs = weight_ptr + (pid_n // head_dim) * weight_stride_0 + (pid_n % head_dim) * weight_stride_1 + k_offsets_fixed[:, None] * weight_stride_2
        weight = tl.load(weight_ptrs, mask=k_mask_fixed[:, None], other=0.0)
        
        # Accumulate
        acc += tl.dot(x, weight.to(tl.float32))
    
    # Store results to Q, K, V tensors
    # Store Q (batch_size, seq_len, num_heads, head_dim)
    q_ptrs = q_ptr + (pid_n // head_dim) * q_stride_0 + m_offsets[:, None] * q_stride_1 + (pid_n % head_dim) * q_stride_2 + k_offsets[None, :] * q_stride_3
    q = acc.to(tl.float32)
    tl.store(q_ptrs, q, mask=m_mask[:, None] & n_mask[None, :])
    
    # Store K (batch_size, seq_len, num_heads, head_dim) - will be transposed later
    k_ptrs = k_ptr + (pid_n // head_dim) * k_stride_0 + m_offsets[:, None] * k_stride_1 + (pid_n % head_dim) * k_stride_2 + k_offsets[None, :] * k_stride_3
    tl.store(k_ptrs, q, mask=m_mask[:, None] & n_mask[None, :])
    
    # Store V (batch_size, seq_len, num_heads, head_dim)
    v_ptrs = v_ptr + (pid_n // head_dim) * v_stride_0 + m_offsets[:, None] * v_stride_1 + (pid_n % head_dim) * v_stride_2 + k_offsets[None, :] * v_stride_3
    tl.store(v_ptrs, q, mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def fused_qkv_kernel_bfloat16(
    weight_ptr,
    x_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    num_heads,
    seq_len,
    head_dim,
    batch_size,
    weight_stride_0,
    weight_stride_1,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    q_stride_0,
    q_stride_1,
    q_stride_2,
    q_stride_3,
    k_stride_0,
    k_stride_1,
    k_stride_2,
    k_stride_3,
    v_stride_0,
    v_stride_1,
    v_stride_2,
    v_stride_3,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Use bfloat16 for better performance on supported GPUs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    k_start = pid_k * BLOCK_K
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    
    m_mask = m_offsets < seq_len
    n_mask = n_offsets < (num_heads * head_dim)
    k_mask = k_offsets < x.size(2)
    
    weight_ptrs = weight_ptr + (pid_n // head_dim) * weight_stride_0 + (pid_n % head_dim) * weight_stride_1 + k_offsets[:, None] * weight_stride_2
    weight = tl.load(weight_ptrs, mask=k_mask[:, None], other=0.0)
    
    x_ptrs = x_ptr + m_offsets[:, None] * x_stride_0 + k_offsets[None, :] * x_stride_2
    x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    for k in range(0, x.size(2), BLOCK_K):
        k_offsets_fixed = k + tl.arange(0, BLOCK_K)
        k_mask_fixed = k_offsets_fixed < x.size(2)
        
        x_ptrs = x_ptr + m_offsets[:, None] * x_stride_0 + k_offsets_fixed[None, :] * x_stride_2
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask_fixed[None, :], other=0.0)
        
        weight_ptrs = weight_ptr + (pid_n // head_dim) * weight_stride_0 + (pid_n % head_dim) * weight_stride_1 + k_offsets_fixed[:, None] * weight_stride_2
        weight = tl.load(weight_ptrs, mask=k_mask_fixed[:, None], other=0.0)
        
        acc += tl.dot(x, weight.to(tl.bfloat16))
    
    q = acc.to(tl.bfloat16)
    
    q_ptrs = q_ptr + (pid_n // head_dim) * q_stride_0 + m_offsets[:, None] * q_stride_1 + (pid_n % head_dim) * q_stride_2 + k_offsets[None, :] * q_stride_3
    tl.store(q_ptrs, q, mask=m_mask[:, None] & n_mask[None, :])
    
    k_ptrs = k_ptr + (pid_n // head_dim) * k_stride_0 + m_offsets[:, None] * k_stride_1 + (pid_n % head_dim) * k_stride_2 + k_offsets[None, :] * k_stride_3
    tl.store(k_ptrs, q, mask=m_mask[:, None] & n_mask[None, :])
    
    v_ptrs = v_ptr + (pid_n // head_dim) * v_stride_0 + m_offsets[:, None] * v_stride_1 + (pid_n % head_dim) * v_stride_2 + k_offsets[None, :] * v_stride_3
    tl.store(v_ptrs, q, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def fused_qkv_linear(weight, x):
    # Extract tensor shapes and properties
    dtype = x.dtype
    batch_size = x.size(0)
    seq_len = x.size(1)
    in_features = x.size(2)
    
    # QKV dimension
    out_features = weight.size(0)
    num_heads = out_features // (3 * 48)
    head_dim = 48
    
    # Determine which kernel to use based on data type
    weight = weight.cuda()
    
    # Create output tensors on GPU
    q = torch.empty((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=x.device)
    k = torch.empty((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=x.device)
    v = torch.empty((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=x.device)
    
    # Reshape weight to (3, num_heads, head_dim, in_features)
    weight_reshaped = weight.view(3, num_heads, head_dim, -1)
    
    # Block sizes - tuned for GPU occupancy
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Calculate grid
    grid_z = num_heads * head_dim
    grid_y = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid_x = (in_features + BLOCK_K - 1) // BLOCK_K
    
    # Choose kernel based on dtype
    if dtype == torch.bfloat16:
        fused_qkv_kernel_bfloat16[(grid_y, grid_z, grid_x)](
            weight_reshaped,
            x,
            q, k, v,
            num_heads,
            seq_len,
            head_dim,
            batch_size,
            weight_reshaped.stride(0),
            weight_reshaped.stride(1),
            weight_reshaped.stride(2),
            weight_reshaped.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            BLOCK_M,
            BLOCK_N,
            BLOCK_K
        )
    else:
        fused_qkv_kernel[(grid_y, grid_z, grid_x)](
            weight_reshaped,
            x,
            q, k, v,
            num_heads,
            seq_len,
            head_dim,
            batch_size,
            weight_reshaped.stride(0),
            weight_reshaped.stride(1),
            weight_reshaped.stride(2),
            weight_reshaped.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            BLOCK_M,
            BLOCK_N,
            BLOCK_K
        )
    
    # Transpose K tensor
    k_transposed = k.transpose(-2, -1)
    
    return q, k_transposed, v

def replacement_func():
    return fused_qkv_linear