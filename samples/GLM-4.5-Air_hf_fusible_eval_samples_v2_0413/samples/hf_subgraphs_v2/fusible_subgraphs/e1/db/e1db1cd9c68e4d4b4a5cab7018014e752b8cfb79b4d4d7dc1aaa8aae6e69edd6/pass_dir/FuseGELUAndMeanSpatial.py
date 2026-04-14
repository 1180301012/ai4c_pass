import torch
import triton
import triton.language as tl

# Pattern matching function - matches GELU + Mean reduction pattern
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized GELU kernel
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    hidden_dim,
    spatial_height,
    spatial_width,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    # Calculate program IDs
    batch_idx = tl.program_id(0)
    hidden_idx = tl.program_id(1)
    spatial_idx = tl.program_id(2)
    
    # Bounds checking
    if batch_idx >= batch_size or hidden_idx >= hidden_dim:
        return
    
    # Calculate spatial dimension
    spatial_dim = spatial_height * spatial_width
    
    # Base pointers
    x_offset = (batch_idx * hidden_dim + hidden_idx) * spatial_dim
    out_offset = (batch_idx * hidden_dim + hidden_idx) * spatial_dim
    
    # Process spatial elements
    spatial_start = spatial_idx * BLOCK_SPATIAL
    spatial_end = min(spatial_start + BLOCK_SPATIAL, spatial_dim)
    spatial_offsets = spatial_start + tl.arange(0, spatial_end - spatial_start)
    spatial_mask = spatial_offsets < spatial_dim
    
    # Load input data
    x_vals = tl.load(x_ptr + x_offset + spatial_offsets, mask=spatial_mask, other=0.0)
    
    # Apply GELU activation
    gelu_vals = 0.5 * x_vals * (1.0 + tl.libdevice.tanh(x_vals * (0.7978845608028654 + 0.044715 * x_vals * x_vals)))
    
    # Store output
    tl.store(out_ptr + out_offset + spatial_offsets, gelu_vals, mask=spatial_mask)

# Optimized mean computation kernel
@triton.jit
def spatial_mean_kernel(
    gelu_ptr,
    out_ptr,
    batch_size,
    hidden_dim,
    spatial_dim,
    BLOCK_HIDDEN: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    hidden_idx = tl.program_id(1)
    
    # Bounds checking
    if batch_idx >= batch_size or hidden_idx >= hidden_dim:
        return
    
    # Load GELU data and compute mean
    base_offset = (batch_idx * hidden_dim + hidden_idx) * spatial_dim
    
    # Sum all spatial elements
    total_sum = 0.0
    
    # Process in blocks for better memory access
    for spatial_block in range(0, spatial_dim, 256):
        block_start = spatial_block
        block_end = min(spatial_block + 256, spatial_dim)
        block_size = block_end - block_start
        
        spatial_offsets = block_start + tl.arange(0, block_size)
        spatial_mask = spatial_offsets < spatial_dim
        
        gelu_vals = tl.load(gelu_ptr + base_offset + spatial_offsets, mask=spatial_mask, other=0.0)
        total_sum += tl.sum(gelu_vals)
    
    # Compute mean
    mean_val = total_sum / spatial_dim
    output_offset = batch_idx * hidden_dim + hidden_idx
    tl.store(out_ptr + output_offset, mean_val)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_gelu_mean_wrapper(x):
    batch_size, hidden_dim, spatial_height, spatial_width = x.shape
    spatial_dim = spatial_height * spatial_width
    
    # Create output tensors
    gelu_out = torch.empty_like(x)
    mean_out = torch.empty(batch_size, hidden_dim, device=x.device, dtype=x.dtype)
    
    # Grid configurations
    BLOCK_HIDDEN = 64
    BLOCK_SPATIAL = 256
    
    # GELU kernel grid
    grid_gelu = (
        batch_size,
        (hidden_dim + BLOCK_HIDDEN - 1) // BLOCK_HIDDEN,
        (spatial_dim + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL
    )
    
    # Mean computation grid (no spatial dimension needed)
    grid_mean = (
        batch_size,
        (hidden_dim + BLOCK_HIDDEN - 1) // BLOCK_HIDDEN,
    )
    
    # Launch GELU kernel
    gelu_kernel[grid_gelu](
        x_ptr=x,
        out_ptr=gelu_out,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        spatial_height=spatial_height,
        spatial_width=spatial_width,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )
    
    # Launch mean computation kernel
    spatial_mean_kernel[grid_mean](
        gelu_ptr=gelu_out,
        out_ptr=mean_out,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        spatial_dim=spatial_dim,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
    )
    
    return gelu_out, mean_out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_gelu_mean_wrapper