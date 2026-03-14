import torch
import triton
import triton.language as tl

def pattern(tmp_1, tmp_0, in_2):
    """Match the fused normalization pattern: scale * relu(input) + bias"""
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = tmp_1 * tmp_2
    tmp_4 = tmp_3 + tmp_0
    return tmp_4

def replacement_args(tmp_1, tmp_0, in_2):
    """Extract arguments for the fused kernel"""
    return (tmp_1, tmp_0, in_2)

@triton.jit
def fused_norm_relu_kernel(
    x_ptr,           # input feature map
    scale_ptr,       # scale factor  
    bias_ptr,        # bias
    out_ptr,         # output
    N,               # batch size
    C,               # channels
    H,               # height
    W,               # width
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: scale * relu(input) + bias"""
    # Calculate program indices
    batch = tl.program_id(0)
    channel = tl.program_id(1)
    
    # Offset for current batch and channel
    offset = batch * C * H * W + channel * H * W
    
    # Load scale and bias (scalar broadcast)
    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    
    # Calculate element offsets within this batch/channel
    elem_idx = tl.arange(0, BLOCK_SIZE)
    elem_offset = offset + elem_idx
    
    # Create mask for valid elements
    mask = elem_offset < N * C * H * W
    
    # Load input values
    x = tl.load(x_ptr + elem_offset, mask=mask, other=0.0)
    
    # Compute: scale * relu(x) + bias
    relu_x = tl.maximum(x, 0.0)
    scale_relu = scale * relu_x
    out = scale_relu + bias
    
    # Store result
    tl.store(out_ptr + elem_offset, out, mask=mask)

@triton.jit
def optimized_large_grid_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel with large grid for better GPU occupancy"""
    # Each program handles one element
    pid = tl.program_id(0)
    
    # Calculate total elements
    total_elements = N * C * H * W
    
    # Create offset and mask
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load scale and bias (broadcast)
    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation
    relu_x = tl.maximum(x, 0.0)
    scale_relu = scale * relu_x
    out = scale_relu + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_norm_relu_triton(scale, bias, x):
    """PyTorch wrapper for the fused kernel"""
    # Get tensor shapes
    N, C, H, W = x.shape
    
    # Set block size based on tensor size
    if x.numel() < 8192:  # Small tensors
        BLOCK_SIZE = 128
    else:  # Large tensors  
        BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    total_elements = N * C * H * W
    if(total_elements > 1000000):
        # Use large grid for better occupancy on large tensors
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (num_programs,)
        
        # Use large grid kernel
        out = torch.empty_like(x)
        optimized_large_grid_kernel[grid](
            x_ptr=x,
            scale_ptr=scale,
            bias_ptr=bias, 
            out_ptr=out,
            N=N, C=C, H=H, W=W,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Use 3D grid for smaller tensors
        grid = (N, C, (H * W + BLOCK_SIZE - 1) // BLOCK_SIZE)
        
        # Use 3D grid kernel
        out = torch.empty_like(x)
        fused_norm_relu_kernel[grid](
            x_ptr=x,
            scale_ptr=scale,
            bias_ptr=bias,
            out_ptr=out,
            N=N, C=C, H=H, W=W,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return out

def replacement_func():
    """Return the fused function"""
    return fused_norm_relu_triton