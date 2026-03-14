import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match element-wise multiply + sum along dim 1 + unsqueeze + sigmoid
    Output structure must exactly match the original model's return
    """
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_0 = None
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_1 = None
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_2 = None
    return tmp_3

def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement kernel"""
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    x_ptr,           # First input tensor [B, C, H, W]
    y_ptr,           # Second input tensor [B, C, H, W]
    out_ptr,         # Output tensor [B, 1, H, W]
    n_elements,      # Number of elements (B * H * W)
    c_elements,      # Number of channels (C)
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Optimized fused kernel that performs:
    1. Element-wise multiplication 
    2. Sum along channel dimension (dim=1)
    3. Apply sigmoid activation
    4. Output has shape [B, 1, H, W]

    Simplified and efficient design:
    - Each program handles one batch and one spatial location (h,w)
    - Load all 64 channels efficiently  
    - Simple element-wise ops + sum
    - No complex reduction needed
    """
    # Get batch index and flattened spatial index
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # For C=64, use multiple threads per program in warp
    # C channels per spatial location, need 64 total threads
    thread_id = tl.program_id(2)
    channel_idx = thread_id * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    
    # Create mask for valid channels (only up to c_elements=64)
    channel_mask = channel_idx < c_elements
    
    # Calculate spatial coordinates from flattened spatial_idx
    h = spatial_idx // 64  # W = 64  
    w = spatial_idx % 64   # W = 64
    
    # Simplified memory addressing for contiguous layout
    # For shape [B, C, H, W], the stride from (h,w) to next (h,w) is C=64
    offset = spatial_idx * c_elements
    x_ptrs = x_ptr + batch_idx * (c_elements * 4096) + offset + channel_idx
    y_ptrs = y_ptr + batch_idx * (c_elements * 4096) + offset + channel_idx
    
    # Load input data
    x = tl.load(x_ptrs, mask=channel_mask, other=0.0)
    y = tl.load(y_ptrs, mask=channel_mask, other=0.0)
    
    # Simple multiply and accumulate using Triton's built-in sum
    product = x * y
    total = tl.sum(product, axis=0)
    
    # Apply sigmoid activation
    sigmoid_result = 1.0 / (1.0 + tl.exp(-total))
    
    # Store result - output has shape [B, 1, H, W] so spatial dimension flattened
    out_offset = batch_idx * 4096 + spatial_idx
    tl.store(out_ptr + out_offset, sigmoid_result)

@torch.fx.wrap
def fused_multiply_sum_sigmoid(x, y):
    """Wrapper function to launch the fused kernel"""
    B, C, H, W = x.shape
    
    # Ensure both tensors have same shape
    assert x.shape == y.shape, f"Input tensors must have same shape, got {x.shape} and {y.shape}"
    assert B > 0 and C > 0 and H > 0 and W > 0, "All dimensions must be positive"
    
    # Create output tensor [B, 1, H, W]
    out = torch.empty((B, 1, H, W), device=x.device, dtype=x.dtype)
    
    # Block size for channels - use 32 threads per block for good warp utilization
    BLOCK_SIZE_C = 32  # Process 32 channels per thread group
    
    # Grid configuration: [batch, spatial_locations, channel_blocks]
    # Each program handles one batch and one spatial location, with multiple threads per spatial location
    spatial_locations = H * W  # 4096 for our case
    n_channel_blocks = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C  # 2 for C=64, BLOCK_SIZE_C=32
    grid = (B, spatial_locations, n_channel_blocks)
    
    # Calculate total elements for validation (B * H * W)
    n_elements = B * H * W
    
    # Launch kernel
    fused_kernel[grid](
        x_ptr=x,
        y_ptr=y, 
        out_ptr=out,
        n_elements=n_elements,
        c_elements=C,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return out

def replacement_func():
    """Return the fused function"""
    return fused_multiply_sum_sigmoid