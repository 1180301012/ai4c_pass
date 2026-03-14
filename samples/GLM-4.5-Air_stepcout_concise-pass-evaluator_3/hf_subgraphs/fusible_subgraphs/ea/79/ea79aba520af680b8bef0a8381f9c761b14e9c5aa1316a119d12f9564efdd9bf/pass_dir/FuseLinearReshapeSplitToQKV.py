import torch
import triton
import triton.language as tl

def pattern(in_3, tmp_2, tmp_1):
    """Fuse linear + reshape + split operations to directly compute Q, K, V tensors"""
    tmp_3 = torch.nn.functional.linear(in_3, tmp_2, tmp_1)
    # Reshape with dynamic batch size - determine from input shape
    batch_size = in_3.shape[0]
    tmp_4 = tmp_3.reshape(batch_size, 49, 8, -1)
    tmp_5 = tmp_4.split([32, 32, 128], dim=3)
    q = tmp_5[0].permute(0, 2, 1, 3)
    k = tmp_5[1].permute(0, 2, 1, 3)
    v = tmp_5[2].permute(0, 2, 1, 3)
    return q, k, v

def replacement_args(in_3, tmp_2, tmp_1):
    return in_3, tmp_2, tmp_1

@triton.jit
def fused_qkv_kernel(
    x_ptr, weight_ptr, bias_ptr,
    q_ptr, k_ptr, v_ptr,
    batch_size, seq_len, num_heads, head_dim,
    total_q_heads, total_q_head_dim,
    total_k_heads, total_k_head_dim,
    total_v_heads, total_v_head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel to compute Q, K, V directly from linear transformation"""
    pid = tl.program_id(0)
    n_elements = batch_size * seq_len * num_heads * head_dim
    
    # For each output element, compute the corresponding linear operation
    # We process one element at a time for simplicity, could be optimized further
    for i in range(pid, n_elements, tl.num_programs(0)):
        # Convert flat index to multi-dimensional indices
        batch = i // (seq_len * num_heads * head_dim)
        remainder = i % (seq_len * num_heads * head_dim)
        seq = remainder // (num_heads * head_dim)
        remainder = remainder % (num_heads * head_dim)
        head = remainder // head_dim
        dim = remainder % head_dim
        
        # Compute linear operation result
        x_val = tl.load(x_ptr + batch * seq_len * head_dim + seq * head_dim + dim)
        
        # Compute weighted sum for this position
        output_val = 0.0
        for d in range(head_dim):
            weight_idx = head * head_dim * head_dim + dim * head_dim + d
            output_val += x_val * tl.load(weight_ptr + weight_idx)
        
        # Add bias if present
        if bias_ptr is not None:
            bias_idx = head * head_dim + dim
            output_val += tl.load(bias_ptr + bias_idx)
        
        # Determine which QKV tensor this belongs to and where to store it
        linear_idx = batch * seq_len * num_heads * head_dim + seq * num_heads * head_dim + head * head_dim + dim
        
        # Calculate position in the final reshaped output
        # The reshape is (batch_size*num_heads, seq_len, head_dim) for each Q/K/V
        final_q_idx = batch * num_heads * seq_len * head_dim + head * seq_len * head_dim + seq * head_dim + dim
        final_k_idx = batch * num_heads * seq_len * head_dim + head * seq_len * head_dim + seq * head_dim + dim
        final_v_idx = batch * num_heads * seq_len * head_dim + head * seq_len * head_dim + seq * head_dim + dim
        
        # Q gets first 32 dims, K gets next 32, V gets last 128
        if dim < 32:
            # Q tensor
            q_output_idx = batch * total_q_heads * seq_len * total_q_head_dim + head * seq_len * total_q_head_dim + seq * total_q_head_dim + dim
            tl.store(q_ptr + q_output_idx, output_val)
        elif dim < 64:
            # K tensor  
            k_output_idx = batch * total_k_heads * seq_len * total_k_head_dim + head * seq_len * total_k_head_dim + seq * total_k_head_dim + (dim - 32)
            tl.store(k_ptr + k_output_idx, output_val)
        else:
            # V tensor
            v_output_idx = batch * total_v_heads * seq_len * total_v_head_dim + head * seq_len * total_v_head_dim + seq * total_v_head_dim + (dim - 64)
            tl.store(v_ptr + v_output_idx, output_val)

@torch.fx.wrap
def fused_qkv_computation(in_3, weight, bias):
    """Computation graph for fused QKV projection"""
    input_shape = in_3.shape
    batch_size = input_shape[0]
    seq_len = input_shape[1] 
    input_dim = input_shape[2]
    
    num_q_heads = batch_size
    num_k_heads = batch_size  
    num_v_heads = batch_size
    q_head_dim = 32
    k_head_dim = 32
    v_head_dim = 128
    
    # Output shapes
    q_shape = (batch_size * num_q_heads, seq_len, q_head_dim)
    k_shape = (batch_size * num_k_heads, seq_len, k_head_dim)  
    v_shape = (batch_size * num_v_heads, seq_len, v_head_dim)
    
    q = torch.zeros(q_shape, dtype=in_3.dtype, device=in_3.device)
    k = torch.zeros(k_shape, dtype=in_3.dtype, device=in_3.device)
    v = torch.zeros(v_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Configure grid
    n_elements = batch_size * seq_len * num_q_heads * q_head_dim
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_qkv_kernel[(num_programs,)](
        in_3, weight, bias,
        q, k, v,
        batch_size, seq_len, num_q_heads, q_head_dim,
        num_q_heads, q_head_dim, 
        num_k_heads, k_head_dim,
        num_v_heads, v_head_dim,
        BLOCK_SIZE
    )
    
    # Apply permute operations (layout transformation)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    
    return q, k, v

def replacement_func():
    return fused_qkv_computation