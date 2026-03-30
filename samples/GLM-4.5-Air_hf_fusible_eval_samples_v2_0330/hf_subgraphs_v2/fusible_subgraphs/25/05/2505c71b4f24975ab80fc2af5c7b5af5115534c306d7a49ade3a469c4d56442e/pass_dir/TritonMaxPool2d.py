import torch
import triton
import triton.language as tl

# Pattern matching function for max_pool2d
def pattern(in_3):
    """Match the max_pool2d operation with specific parameters"""
    tmp_5 = torch.nn.functional.max_pool2d(in_3, 2, 1, 0, 1, ceil_mode=True, return_indices=False)
    return tmp_5

# Argument extraction function
def replacement_args(in_3):
    return (in_3,)

# Optimized max pool2d kernel
@triton.jit
def max_pool2d_kernel(
    x_ptr,           # input tensor pointer
    out_ptr,         # output tensor pointer  
    N, C, IH, IW,    # input dimensions: Batch, Channels, Height, Width
    OH, OW,          # output dimensions
    KH, KW,          # kernel height and width
    SH, SW,          # stride height and width
    PH, PW,          # padding height and width
    DH, DW,          # dilation height and width
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    
    # Calculate batch offset
    batch_offset = batch_id * C * IH * IW
    
    # Calculate output coordinates
    in_y_start = out_y * SH - PH
    in_x_start = out_x * SW - PW
    
    # Calculate output index
    out_idx = batch_offset + channel_id * OH * OW + out_y * OW + out_x
    
    # Initialize output with minimum values
    max_val = -tl.float32('inf')
    
    # Iterate over kernel window
    for ky in range(0, KH):
        for kx in range(0, KW):
            # Calculate input coordinates with padding
            in_y = in_y_start + ky * DH
            in_x = in_x_start + kx * DW
            
            # Check bounds
            if (0 <= in_y < IH) and (0 <= in_x < IW):
                # Calculate input index
                in_idx = batch_offset + channel_id * IH * IW + in_y * IW + in_x
                
                # Load input value
                val = tl.load(x_ptr + in_idx, other=-tl.float32('inf'))
                # Update max value
                if val > max_val:
                    max_val = val
    
    # Store output
    tl.store(out_ptr + out_idx, max_val)

@torch.fx.wrap
def triton_max_pool2d(x):
    """Triton max_pool2d wrapper function"""
    input_shape = x.shape
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D input, got {len(input_shape)}D")
    
    # Input dimensions
    N, C, IH, IW = input_shape
    
    # Pooling parameters (matching the target pattern)
    KH, KW = 2, 2  # kernel_size=2
    SH, SW = 1, 1  # stride=1
    PH, PW = 0, 0  # padding=0
    DH, DW = 1, 1  # dilation=1
    
    # Calculate output dimensions with ceil_mode=True
    OH = (IH + 2 * PH - DH * (KH - 1) - 1) // SH + 1
    OW = (IW + 2 * PW - DW * (KW - 1) - 1) // SW + 1
    
    # Ensure we don't go smaller than 1x1 in case of extreme cases
    OH = max(1, OH)
    OW = max(1, OW)
    
    # Create output tensor
    out = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)
    
    # Determine block size and grid dimensions
    BLOCK_SIZE = 64  # Adjust based on performance
    
    # Grid: (batch, channels, output_y, output_x)  
    grid = (N, C, OH, OW)
    
    # Launch kernel
    max_pool2d_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        N=N, C=C, IH=IH, IW=IW,
        OH=OH, OW=OW,
        KH=KH, KW=KW,
        SH=SH, SW=SW,
        PH=PH, PW=PW,
        DH=DH, DW=DW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return triton_max_pool2d