import torch
import triton
import triton.language as tl
import math

def pattern(in_0):
    tmp_1 = in_0.transpose(-2, -1)
    return tmp_1

# Debug: add a simple pattern check
def debug_pattern():
    import torch
    # Test with sample data
    test_tensor = torch.randn(2, 1, 4, 3, device='cuda', dtype=torch.float16)
    result = test_tensor.transpose(-2, -1)
    print(f"Debug: Input shape: {test_tensor.shape}")
    print(f"Debug: Output shape: {result.shape}")
    return result

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def transpose_slice_kernel(
    x_ptr,
    out_ptr,
    h, w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of elements in the 2D slice
    pid = tl.program_id(0)
    n_elements = h * w
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert 1D offset to 2D coordinates
    i = offsets // w
    j = offsets % w
    
    # Create coordinate pairs for transpose
    coords_i = i[:, None]
    coords_j = j[None, :]
    
    # Flatten coordinates for direct memory access
    flat_coords = coords_i * w + coords_j
    
    # Load original data using mask
    x = tl.load(x_ptr + flat_coords, mask=mask, other=0.0)
    
    # Calculate transposed positions
    transpose_coords = coords_j * w + coords_i
    transpose_flat = transpose_coords
    
    # Store in transposed positions
    tl.store(out_ptr + transpose_flat, x, mask=mask)

@torch.fx.wrap  
def optimized_transpose(x):
    # Input shape: [batch, channels, height, width]
    batch_size, channels, height, width = x.shape
    
    # Create output tensor with transposed shape [batch, channels, width, height]
    output_shape = [batch_size, channels, width, height]
    result = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # For each (batch, channel) slice, transpose the 2D matrix
    for b in range(batch_size):
        for c in range(channels):
            # Extract 2D slice to transpose [height, width] -> [width, height]
            input_slice = x[b, c]
            output_slice = result[b, c]
            
            # Choose optimal BLOCK_SIZE for this slice
            slice_elements = height * width
            
            if slice_elements < 1024:
                BLOCK_SIZE = 256
            elif slice_elements < 8192:
                BLOCK_SIZE = 512
            elif slice_elements < 65536:
                BLOCK_SIZE = 1024
            else:
                BLOCK_SIZE = 2048
                
            num_programs = (slice_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            # Launch Triton kernel for this slice
            transpose_slice_kernel[(num_programs,)](
                x_ptr=input_slice,
                out_ptr=output_slice,
                h=height, w=width,
                BLOCK_SIZE=BLOCK_SIZE
            )
    
    return result

def replacement_func():
    # This returns a function that takes (in_0,) and returns tmp_1 (the transposed result)
    def wrapper(in_0):
        return optimized_transpose(in_0)
    return wrapper