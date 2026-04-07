import torch
import triton
import triton.language as tl

# Pattern matching function for input reshape + transpose
def pattern(in_0):
    # This targets the main computation path: input reshape + transpose
    # in_0: [1, 133, 133, 96] 
    # tmp_5: [1, 19, 7, 19, 7, 96]
    # tmp_6: [1, 19, 19, 7, 7, 96] (after transpose(2,3))
    
    # Spatial decomposition: 133 = 19*7
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, in_0.shape[-1])
    tmp_6 = tmp_5.transpose(2, 3)
    return tmp_6

# Triton kernel for optimized reshape + transpose
@triton.jit
def input_reshape_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    h_total, w_total, channels,
    h1, w1, h2, w2,  # Spatial decomposition: 19, 7, 19, 7
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles one block
    h_idx = tl.program_id(0)
    w_idx = tl.program_id(1)
    c_idx = tl.program_id(2)
    
    # Compute offset ranges for this block
    start_h = h_idx * BLOCK_SIZE_H
    start_w = w_idx * BLOCK_SIZE_W
    start_c = c_idx * BLOCK_SIZE_C
    
    # Create offset arrays
    h_offsets = start_h + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = start_w + tl.arange(0, BLOCK_SIZE_W)
    c_offsets = start_c + tl.arange(0, BLOCK_SIZE_C)
    
    # Create masks for boundaries
    mask_h = h_offsets < h1
    mask_w = w_offsets < w2  # Transposed dimension
    mask_c = c_offsets < channels
    
    # Compute input coordinates: [batch, h_total, w_total, channels]
    # We need to decompose h_total = h1*w1*h2*w2 = 19*7*19*7
    # and w_total is not used in the final output for this pattern
    
    # The reshape creates: [batch, h1, w1, h2, w2, channels]
    # The transpose(2,3) swaps w1 and h2: [batch, h1, h2, w2, w1, channels]
    
    # For simplicity, let's create a more direct mapping
    # Input: [batch, h_total, w_total, channels] where h_total=133, w_total=133
    # Output: [batch, h1*h2, w1*w2, channels] where h1=19, h2=19, w1=7, w2=7
    # This represents the spatial decomposition after transpose
    
    # Calculate input indices
    # h_idx_out = h_orig * h2 + w_orig  where h_orig < h1, w_orig < h2
    # w_idx_out = w1_orig * w2 + w2_orig where w1_orig < w1, w2_orig < w2
    
    # Direct computation for better memory access patterns
    output_coords_h = h_offsets
    output_coords_w = w_offsets  
    batch_coords = tl.arange(0, batch_size)[:, None, None, None]
    
    # Convert back to input coordinates
    # h1_orig = output_coords_h // h2
    # w_orig = output_coords_h % h2
    # w1_orig = output_coords_w // w2  
    # w2_orig = output_coords_w % w2
    
    # Flatten for input access: h_total = h1 * w1 * h2 * w2
    input_h = output_coords_h * w1 * w2 * channels
    input_w = output_coords_w * channels
    input_coords = input_h[:, None] + input_w[None, :]
    
    # Load data with proper broadcasting over batch and channels
    input_data = tl.load(input_ptr + 
                        (batch_coords[:, :, None, None] * h_total * w_total * channels +
                         input_coords[:, None, :, None] * w_total +
                         c_offsets[None, None, None, :]),
                        mask=(mask_h[:, None] & mask_w[None, :] & mask_c[None, None, :]),
                        other=0.0).to(tl.float32)
    
    # Store output in transposed layout
    output_coords = (output_coords_h[:, None] * w2 * channels + 
                     output_coords_w[None, :] * channels +
                     c_offsets[None, None, :])
    
    tl.store(output_ptr + (batch_coords[:, :, None, None] * h1 * h2 * w1 * w2 + output_coords[:, None, :, None] + c_offsets[None, None, None, :]),
              input_data,
              mask=(mask_h[:, None] & mask_w[None, :] & mask_c[None, None, :]))

# Kernel wrapper
@torch.fx.wrap
def optimized_input_reshape(in_0):
    batch_size, h_total, w_total, channels = in_0.shape
    
    # Spatial decomposition constants
    h1, w1, h2, w2 = 19, 7, 19, 7
    
    # Create output tensor in transposed layout
    output_shape = (batch_size, h1 * h2, w1 * w2, channels)  # [1, 19*19, 7*7, 96] = [1, 361, 49, 96]
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Set up grid dimensions
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    
    grid_h = (h1 * h2 + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (w1 * w2 + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    input_reshape_transpose_kernel[(grid_h, grid_w, grid_c)](
        input_ptr=in_0,
        output_ptr=out,
        batch_size=batch_size,
        h_total=h_total,
        w_total=w_total,
        channels=channels,
        h1=h1, w1=w1, h2=h2, w2=w2,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Replacement function
def replacement_func():
    return optimized_input_reshape