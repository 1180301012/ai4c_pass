import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern: ReLU(in_1) + in_0 followed by global average pooling (adaptive_avg_pool2d with output_size=1)"""
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2

def replacement_args(in_0, in_1):
    """Extract arguments for the fused operation"""
    return (in_0, in_1)

@triton.jit
def fused_relu_add_global_avg_pool_kernel(
    x_ptr,           # in_0 pointer  
    y_ptr,           # in_1 pointer
    out_ptr,         # output pointer
    batch_size,
    channels,
    height,
    width,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr
):
    """Fused kernel: ReLU(y) + x followed by global average pooling"""
    
    # Calculate output position - global pooling reduces to [B, C, 1, 1]
    c = tl.program_id(0)
    b = tl.program_id(1)
    
    # Pointers for this batch and channel
    x_ptr_bc = x_ptr + b * channels * height * width + c * height * width
    y_ptr_bc = y_ptr + b * channels * height * width + c * height * width
    out_ptr_bc = out_ptr + b * channels + c
    
    # Initialize accumulator for global pooling
    sum_val = 0.0
    
    # Process spatial dimensions with tiling
    for hw_offset in range(0, height * width, BLOCK_HW):
        # Calculate current block bounds
        hw_begin = hw_offset
        hw_end = min(hw_offset + BLOCK_HW, height * width)
        
        # Process elements in this block
        for hw in range(hw_begin, hw_end):
            # Load input elements
            x = tl.load(x_ptr_bc + hw, other=0.0)
            y = tl.load(y_ptr_bc + hw, other=0.0)
            
            # Fused computation: ReLU(y) + x
            relu_y = tl.max(y, 0.0)
            result = relu_y + x
            
            # Accumulate for global pooling
            sum_val += result
    
    # Global average pooling: divide by total spatial elements
    spatial_elements = height * width
    avg_val = sum_val / spatial_elements
    
    # Store result
    tl.store(out_ptr_bc, avg_val)

@torch.fx.wrap
def fused_relu_add_global_avg_pool(in_0, in_1):
    """Wrapper function to launch the fused kernel"""
    B, C, H, W = in_0.shape
    
    # Create output tensor [B, C, 1, 1]
    output = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)
    output_flat = output.view(B, C)  # Flatten spatial dimensions
    
    # Block sizes for tiling
    BLOCK_C = 64
    BLOCK_HW = 256  # Process multiple spatial elements per thread
    
    # Calculate grid dimensions
    grid_c = (C + BLOCK_C - 1) // BLOCK_C
    grid_b = B
    
    # Launch kernel
    fused_relu_add_global_avg_pool_kernel[grid_c, grid_b, (
        BLOCK_C,
        BLOCK_HW
    )](
        in_0,
        in_1,
        output_flat,
        B, C, H, W,
        BLOCK_C,
        BLOCK_HW
    )
    
    return output

def replacement_func():
    """Return the fused function"""
    return fused_relu_add_global_avg_pool