import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matching for QKV projection with transpose (convit_tiny variant).
    Matches: linear -> reshape -> permute -> unbind -> transpose
    """
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = tmp_1.reshape(1, 197, 3, 4, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    tmp_4 = tmp_3.unbind(0)
    tmp_5 = tmp_4[0]
    tmp_6 = tmp_4[1]
    tmp_7 = tmp_4[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_qkv_kernel(
    input_ptr,
    weight_ptr,
    q_ptr,
    k_t_ptr,
    v_ptr,
    batch_size,
    seq_len,
    in_features,
    num_heads,
    head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused QKV projection kernel.
    Input: [batch_size, seq_len, in_features]
    Weight: [3 * num_heads * head_dim, in_features]
    Outputs:
    - Q: [batch_size, num_heads, seq_len, head_dim]
    - K^T: [batch_size, num_heads, head_dim, seq_len]
    - V: [batch_size, num_heads, seq_len, head_dim]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Compute which output we're working on (Q=0, K=1, V=2)
    qkv_idx = pid_n // num_heads
    head_idx = pid_n % num_heads
    
    # Compute output row (sequence position or head_dim position for K^T)
    row_idx = pid_m
    
    # For each output element, we need to compute a dot product
    # output[row] = input[batch, row // head_dim, :] @ weight[qkv_offset + head_offset + row % head_dim, :]
    
    if qkv_idx == 1:  # K, will be transposed
        # For K^T, we compute K first then write transposed
        # K shape: [batch, num_heads, seq_len, head_dim]
        # K^T shape: [batch, num_heads, head_dim, seq_len]
        # row_idx represents head_dim index in K^T, which is head_dim index in K
        if row_idx >= head_dim:
            return
        
        # Compute for all seq positions
        for seq_idx in range(seq_len):
            # Compute K[batch, head_idx, seq_idx, row_idx]
            acc = tl.zeros([1], dtype=tl.float32)
            
            weight_row = qkv_idx * num_heads * head_dim + head_idx * head_dim + row_idx
            
            for k in range(0, in_features, BLOCK_SIZE_K):
                k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
                k_mask = k_offsets < in_features
                
                # Load input[batch, seq_idx, k_offsets]
                input_offset = pid_batch * seq_len * in_features + seq_idx * in_features + k_offsets
                input_vals = tl.load(input_ptr + input_offset, mask=k_mask, other=0.0)
                
                # Load weight[weight_row, k_offsets]
                weight_offset = weight_row * in_features + k_offsets
                weight_vals = tl.load(weight_ptr + weight_offset, mask=k_mask, other=0.0)
                
                acc += tl.sum(input_vals * weight_vals)
            
            # Write to K^T[batch, head_idx, row_idx, seq_idx]
            k_t_offset = pid_batch * num_heads * head_dim * seq_len + head_idx * head_dim * seq_len + row_idx * seq_len + seq_idx
            tl.store(k_t_ptr + k_t_offset, acc)
    
    else:  # Q or V
        # row_idx represents seq_len index
        if row_idx >= seq_len:
            return
        
        # Compute for all head_dim positions
        for dim_idx in range(head_dim):
            acc = tl.zeros([1], dtype=tl.float32)
            
            weight_row = qkv_idx * num_heads * head_dim + head_idx * head_dim + dim_idx
            
            for k in range(0, in_features, BLOCK_SIZE_K):
                k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
                k_mask = k_offsets < in_features
                
                # Load input[batch, row_idx, k_offsets]
                input_offset = pid_batch * seq_len * in_features + row_idx * in_features + k_offsets
                input_vals = tl.load(input_ptr + input_offset, mask=k_mask, other=0.0)
                
                # Load weight[weight_row, k_offsets]
                weight_offset = weight_row * in_features + k_offsets
                weight_vals = tl.load(weight_ptr + weight_offset, mask=k_mask, other=0.0)
                
                acc += tl.sum(input_vals * weight_vals)
            
            # Write to output
            if qkv_idx == 0:  # Q
                q_offset = pid_batch * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim + row_idx * head_dim + dim_idx
                tl.store(q_ptr + q_offset, acc)
            else:  # V (qkv_idx == 2)
                v_offset = pid_batch * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim + row_idx * head_dim + dim_idx
                tl.store(v_ptr + v_offset, acc)

@torch.fx.wrap
def fused_qkv_projection_tiny(weight, input):
    """
    Fused QKV projection with K transpose (convit_tiny variant: 4 heads).
    """
    batch_size, seq_len, in_features = input.shape
    out_features = weight.shape[0]
    
    # convit_tiny: 4 heads, 48 head_dim
    head_dim = 48
    num_heads = 4
    
    # Allocate outputs
    q = torch.empty(batch_size, num_heads, seq_len, head_dim, device=input.device, dtype=input.dtype)
    k_t = torch.empty(batch_size, num_heads, head_dim, seq_len, device=input.device, dtype=input.dtype)
    v = torch.empty(batch_size, num_heads, seq_len, head_dim, device=input.device, dtype=input.dtype)
    
    # Launch kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 64
    
    # Grid: (max(seq_len, head_dim), 3 * num_heads, batch_size)
    grid_m = max(seq_len, head_dim)
    grid_n = 3 * num_heads
    
    fused_qkv_kernel[(grid_m, grid_n, batch_size)](
        input,
        weight,
        q,
        k_t,
        v,
        batch_size,
        seq_len,
        in_features,
        num_heads,
        head_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return (q, k_t, v)

def replacement_func():
    return fused_qkv_projection_tiny