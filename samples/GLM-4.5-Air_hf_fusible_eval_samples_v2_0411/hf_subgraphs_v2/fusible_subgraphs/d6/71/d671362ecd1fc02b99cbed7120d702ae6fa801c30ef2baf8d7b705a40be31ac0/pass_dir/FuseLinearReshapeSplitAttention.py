import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_reshape_split_kernel(
    x_ptr,           # Input tensor [batch, 49, 448]
    weight_ptr,      # Weight matrix [1536, 448]  
    bias_ptr,        # Bias vector [1536]
    q_out_ptr,       # Output Q tensor [batch, 49, 8, 32]
    k_out_ptr,       # Output K tensor [batch, 49, 8, 32] 
    v_out_ptr,       # Output V tensor [batch, 49, 8, 128]
    batch_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Linear transformation: x @ weight.T + bias
    2. Reshape to [batch, 49, 8, 192] 
    3. Split into Q, K, V components (32, 32, 128)
    """
    pid = tl.program_id(0)
    batch_idx = pid // (49 * 8)
    seq_idx = (pid // 8) % 49
    head_idx = pid % 8
    component_idx = 0  # 0=Q, 1=K, 2=V
    
    # Compute linear output position
    linear_offset = batch_idx * 49 * 448 + seq_idx * 448
    weight_offset = head_idx * 192 * 448 + component_idx * 64 * 448
    
    # Process Q component (first 192//8 = 24 elements per head)
    for k_off in tl.range(0, 448, BLOCK_SIZE_K):
        k_ptr = weight_ptr + weight_offset + k_off
        bias_val = tl.load(bias_ptr + head_idx * 192 + component_idx * 64 + k_off)
        x_val = tl.load(x_ptr + linear_offset + k_off)
        
        # Compute matrix multiplication for this segment
        acc = 0.0
        for k in range(k_off, min(k_off + BLOCK_SIZE_K, 448)):
            w = tl.load(k_ptr + (k - k_off))
            acc += x_val * w
        acc += bias_val
        
        # Store to appropriate output component
        out_offset = batch_idx * 49 * 8 * 192 + seq_idx * 8 * 192 + head_idx * 192 + component_idx * 64 + (k_off - component_idx * 64 * 448)
        if k_off < 32:  # Q component
            tl.store(q_out_ptr + head_idx * 64 * 448 + seq_idx * 64 * 8 + batch_idx * 49 * 64 * 8 + k_off, acc)
        elif k_off < 64:  # K component 
            tl.store(k_out_ptr + head_idx * 64 * 448 + seq_idx * 64 * 8 + batch_idx * 49 * 64 * 8 + (k_off - 32), acc)
        else:  # V component
            tl.store(v_out_ptr + head_idx * 192 * 448 + seq_idx * 192 * 8 + batch_idx * 49 * 192 * 8 + (k_off - 64), acc)

@torch.fx.wrap
def fused_linear_reshape_split(x, weight, bias):
    """Fused linear + reshape + split operation for attention mechanism"""
    batch_size = x.shape[0]
    num_heads = 8
    seq_len = 49
    head_dim_qk = 32
    head_dim_v = 128
    
    # Output tensors
    q_out = torch.empty(batch_size, seq_len, num_heads, head_dim_qk, dtype=x.dtype, device=x.device)
    k_out = torch.empty(batch_size, seq_len, num_heads, head_dim_qk, dtype=x.dtype, device=x.device) 
    v_out = torch.empty(batch_size, seq_len, num_heads, head_dim_v, dtype=x.dtype, device=x.device)
    
    # Calculate grid size
    grid_size = batch_size * seq_len * num_heads
    
    fused_linear_reshape_split_kernel[grid_size](
        x, weight, bias, q_out, k_out, v_out,
        batch_size,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=128, 
        BLOCK_SIZE_K=32
    )
    
    return q_out, k_out, v_out

def pattern(in_3, in_2, in_1):
    """
    Match linear + reshape + split pattern from the computation graph
    """
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    tmp_4 = linear.reshape(1, 49, 8, -1)  # batch_size varies, use the first batch as representative
    split = tmp_4.split([32, 32, 128], dim=3)
    tmp_6 = split[0]
    tmp_7 = split[1]
    tmp_8 = split[2]
    return tmp_6, tmp_7, tmp_8

def replacement_args(in_3, in_2, in_1):
    return (in_3, in_2, in_1)

def replacement_func():
    return fused_linear_reshape_split