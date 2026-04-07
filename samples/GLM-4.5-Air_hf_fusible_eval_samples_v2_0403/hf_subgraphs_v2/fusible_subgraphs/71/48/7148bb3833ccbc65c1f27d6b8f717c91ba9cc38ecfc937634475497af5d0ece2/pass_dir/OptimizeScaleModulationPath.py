import torch
import triton
import triton.language as tl

def pattern(in_0, tmp_4, tmp_2):
    tmp_5 = in_0.unsqueeze(-1)
    tmp_0 = None
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_5 = None
    tmp_7 = tmp_6 * tmp_4
    tmp_6 = tmp_4 = None
    tmp_8 = tmp_2 + tmp_7
    tmp_2 = tmp_7 = None
    return tmp_8

def replacement_args(in_0, tmp_4, tmp_2):
    return (in_0, tmp_4, tmp_2)

@triton.jit
def scale_modulation_kernel(
    scale_ptr,
    diff_ptr,
    relu_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    ids = tl.program_id(0)
    
    # Simple 1D grid for simplicity
    total_elements = N * C * H * W
    if ids >= total_elements:
        return
    
    # Calculate coordinates
    offset = ids
    n = offset // (C * H * W)
    offset = offset % (C * H * W)
    c = offset // (H * W)
    offset = offset % (H * W)
    h = offset // W
    w = offset % W
    
    # Load scale for this channel
    scale = tl.load(scale_ptr + c, other=0.0)
    scale_scaled = scale  # Already in [C], expand in computation
    
    # Load diff and relu features
    diff_idx = n * C * H * W + c * H * W + h * W + w
    relu_idx = n * C * H * W + c * H * W + h * W + w
    
    diff_val = tl.load(diff_ptr + diff_idx, other=0.0)
    relu_val = tl.load(relu_ptr + relu_idx, other=0.0)
    
    # Compute: relu + diff * scale
    result = relu_val + diff_val * scale
    
    # Store result
    output_idx = n * C * H * W + c * H * W + h * W + w
    tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def scale_modulation_optimized_torch(in_0, tmp_4, tmp_2):
    # Get tensor shapes
    N, C, H, W = tmp_2.shape
    
    output = torch.empty_like(tmp_2)
    
    # Calculate total number of elements
    total_elements = N * C * H * W
    
    # Kernel configuration  
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions (1D)
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    scale_modulation_kernel[(num_programs,)](
        scale_ptr=in_0,
        diff_ptr=tmp_4,
        relu_ptr=tmp_2,
        output_ptr=output,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE_M=BLOCK_SIZE,
        BLOCK_SIZE_N=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return scale_modulation_optimized_torch