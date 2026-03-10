import torch
import triton
import triton.language as tl

# Pattern matching function - matches element-wise division + transpose
def pattern(x, scale):
    # tmp_0 = x / scale (element-wise division)
    tmp_0 = x / scale
    # tmp_1 = tmp_0.transpose(-1, -2) 
    tmp_1 = tmp_0.transpose(-1, -2)
    # tmp_0 = None (cleanup, not included in pattern)
    return tmp_1

# Argument extraction function
def replacement_args(x, scale):
    return (x, scale)

# Optimized fused kernel with simple 3D grid for better performance  
@triton.jit
def fused_divide_transpose_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale,
):
    # Simple 3D grid launch with bounds checking
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Check bounds for sequence dimension, divide head_dim into chunks
    seq_end = min(seq_idx + 1, seq_len)
    dim_chunk_size = (head_dim + 2) // 3  # Divide head_dim into 3 chunks
    
    # Process a chunk of head dimensions for this program
    dim_start = (batch_id % 3) * dim_chunk_size
    dim_end = min(dim_start + dim_chunk_size, head_dim)
    
    if batch_id >= 3:  # Only use first grid dimension for multiple chunks
        return
    
    # Process chunk of head dimensions for this sequence position
    for dim_idx in range(dim_start, dim_end):
        # Calculate input pointer: [batch, head, seq, dim]
        input_offset = batch_id * num_heads * seq_len * head_dim + head_id * seq_len * head_dim + seq_idx * head_dim + dim_idx
        
        # Calculate output pointer: [batch, head, dim, seq] (transposed)  
        output_offset = batch_id * num_heads * head_dim * seq_len + head_id * head_dim * seq_len + dim_idx * seq_len + seq_idx
        
        # Load input element, apply division, and store to transposed position
        x_val = tl.load(x_ptr + input_offset)
        result = x_val / scale
        tl.store(out_ptr + output_offset, result)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_divide_transpose(x, scale):
    batch_size, num_heads, seq_len, head_dim = x.shape
    
    # Output shape will have last two dimensions transposed
    out_shape = (batch_size, num_heads, head_dim, seq_len)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel with 3D grid: (3, num_heads, seq_len)
    # We use 3 for batch dimension to process head_dim in chunks
    grid_size_x = 3  # Process head_dim in 3 chunks
    grid_size_y = num_heads
    grid_size_z = seq_len
    
    fused_divide_transpose_kernel[(grid_size_x, grid_size_y, grid_size_z)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        scale=scale
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_divide_transpose