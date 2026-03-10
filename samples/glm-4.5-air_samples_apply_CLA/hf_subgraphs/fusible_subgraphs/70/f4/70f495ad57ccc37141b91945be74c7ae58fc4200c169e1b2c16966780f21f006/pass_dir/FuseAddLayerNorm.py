import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1280,), in_1, in_0, 1e-06)
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_add_layernorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, seq_len, hidden_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Each program handles a block of data
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute memory offsets within the block
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < batch_size
    n_mask = n_offsets < hidden_size
    
    # Load addition result
    x = tl.load(x_ptr + (m_offsets[:, None] * seq_len) * hidden_size + n_offsets[None, :], 
               mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Load operation result  
    y = tl.load(y_ptr + (m_offsets[:, None] * seq_len) * hidden_size + n_offsets[None, :], 
               mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Addition operation
    added = x + y
    
    # Load layer norm parameters
    weight = tl.load(weight_ptr + n_offsets, mask=n_mask, other=1.0)
    bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
    
    # Compute mean with better axis handling
    mean = tl.sum(added, axis=1) / hidden_size
    mean = mean[:, None]
    
    # Compute variance
    diff = added - mean
    var = tl.sum(diff * diff, axis=1) / hidden_size
    var = var[:, None]
    
    # Layer normalization with better numerical stability
    normalized = (added - mean) * tl.rsqrt(var + 1e-06) * weight + bias
    
    # Store result with optimized addressing
    tl.store(out_ptr + (m_offsets[:, None] * seq_len) * hidden_size + n_offsets[None, :], 
            normalized, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap  
def fused_add_layernorm(in_0, in_1, in_2, in_3):
    batch_size, seq_len, hidden_size = in_2.shape
    
    output = torch.empty_like(in_2)
    
    # Optimized block size selection based on workload characteristics
    total_elements = batch_size * hidden_size * seq_len
    
    if total_elements >= 262144:  # Large workload (e.g., batch_size=32, hidden_size=1280)
        BLOCK_M = 8   # Smaller blocks for better parallelism
        BLOCK_N = 256
    elif total_elements >= 65536:  # Medium workload
        BLOCK_M = 16
        BLOCK_N = 128
    elif total_elements >= 16384:  # Small to medium workload
        BLOCK_M = 32
        BLOCK_N = 64
    else:  # Small workload
        BLOCK_M = 64
        BLOCK_N = 32
    
    # Optimize launch grid based on available resources
    grid_dim_m = triton.cdiv(batch_size, BLOCK_M)
    grid_dim_n = triton.cdiv(hidden_size, BLOCK_N)
    
    # Limit grid size to avoid oversubscription
    max_grid_dim = 65535  # Maximum block dimensions per GPU
    grid_dim_m = min(grid_dim_m, max_grid_dim)
    grid_dim_n = min(grid_dim_n, max_grid_dim)
    
    fused_add_layernorm_kernel[(grid_dim_m, grid_dim_n)](
        in_2, in_3, in_1, in_0, output,
        batch_size, seq_len, hidden_size,
        BLOCK_M, BLOCK_N
    )
    
    return output

def replacement_func():
    return fused_add_layernorm