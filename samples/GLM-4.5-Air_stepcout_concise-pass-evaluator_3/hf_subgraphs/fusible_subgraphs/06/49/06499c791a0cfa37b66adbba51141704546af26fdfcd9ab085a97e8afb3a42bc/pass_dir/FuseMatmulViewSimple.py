import torch
import triton
import triton.language as tl

def pattern(a, b):
    """
    Pattern matching matmul operation across all models.
    """
    return a @ b

def replacement_args(a, b):
    """Extract arguments needed for the fused matmul+view operation."""
    return (a, b)

@triton.jit
def simple_matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    batch_size: tl.constexpr,
    a_channels: tl.constexpr, 
    a_feat: tl.constexpr,
    b_feat: tl.constexpr,
    b_last: tl.constexpr,
):
    """
    Simple optimized kernel for matrix multiplication.
    """
    # Program IDs for parallel execution
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    feat_id = tl.program_id(2)
    
    # Check if we're within bounds for this thread
    batch_mask = batch_id < batch_size
    channel_mask = channel_id < a_channels
    feat_mask = feat_id < b_last
    
    # Only proceed if we're within valid bounds
    if not ((batch_mask and channel_mask) and feat_mask):
        return
    
    # Compute strides
    a_stride = a_channels * a_feat * b_feat
    b_stride = a_channels * b_feat * b_last
    out_stride = a_channels * a_feat * b_last
    
    # Calculate output index
    out_offset = batch_id * out_stride + channel_id * a_feat * b_last + feat_id * b_last
    
    # Perform matrix multiplication
    accumulator = 0.0
    for k in range(b_feat):
        a_offset = batch_id * a_stride + channel_id * a_feat * b_feat + k * b_feat + feat_id
        b_offset = batch_id * b_stride + channel_id * b_feat * b_last + k * b_last + feat_id
        
        # Load without bounds checking (assumes valid indices due to earlier mask check)
        a_val = tl.load(a_ptr + a_offset)
        b_val = tl.load(b_ptr + b_offset)
        accumulator += a_val * b_val
    
    # Store result
    tl.store(out_ptr + out_offset, accumulator)

@torch.fx.wrap
def simple_matmul(a, b):
    """
    Wrapper function that launches the optimized matmul kernel.
    """
    # Get input tensor shapes
    batch_size = a.shape[0]
    a_channels = a.shape[1]
    a_feat = a.shape[2]
    b_feat = a.shape[3]
    b_channels = b.shape[1]  # Should match a_channels
    b_last = b.shape[3]      # Last dimension of b
    
    # Create output tensor with matmul result shape
    # For matmul: [B, C_a, N, P] @ [B, C_b, M, P] -> [B, C_a, N, M] when using @ operator
    # But the actual models expect: view(shape after matmul)
    out_shape = (batch_size, a_channels, a_feat, b_last)
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    
    # Calculate grid dimensions
    grid_x = batch_size
    grid_y = a_channels
    grid_z = b_last
    
    # Launch kernel
    simple_matmul_kernel[grid_x, grid_y, grid_z](
        a_ptr=a,
        b_ptr=b, 
        out_ptr=out,
        batch_size=batch_size,
        a_channels=a_channels,
        a_feat=a_feat,
        b_feat=b_feat,
        b_last=b_last,
    )
    
    return out

def replacement_func():
    """
    Returns the optimized function that performs matmul operation.
    """
    return simple_matmul