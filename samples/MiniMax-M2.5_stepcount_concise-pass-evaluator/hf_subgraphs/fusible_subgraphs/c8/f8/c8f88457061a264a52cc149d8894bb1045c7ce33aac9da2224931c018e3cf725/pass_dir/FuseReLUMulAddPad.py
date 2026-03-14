import torch
import triton
import triton.language as tl


# Pattern matching function - matches the exact computation pattern from model.py
# The pattern: ReLU(in_2) -> multiply by in_1 -> add in_0 -> pad
def pattern(in_0, in_1, in_2):
    """
    Define the computation pattern to match.
    This matches: relu(in_2) * in_1 + in_0, then pad
    """
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5


# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Optimized Triton kernel that fuses ReLU, multiply, add, and pad operations
@triton.jit
def fused_relu_mul_add_pad_kernel(
    in_ptr,      # Input tensor (in_2)
    scale_ptr,   # Scale value (in_1) - scalar
    bias_ptr,    # Bias value (in_0) - scalar  
    out_ptr,     # Output tensor
    in_h,        # Input height
    in_w,        # Input width
    out_h,       # Output height (in_h + 1)
    out_w,       # Output width (in_w + 1)
    stride_in,   # Stride for input
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. ReLU activation
    2. Multiply by scale
    3. Add bias
    4. Padding (output is larger than input)
    """
    # Each program handles a 2D tile of output
    pid = tl.program_id(0)
    
    # Calculate the starting position for this program
    # We process in 2D but flatten for now
    num_elements = out_h * out_w
    block_start = pid * BLOCK_SIZE
    
    # Calculate offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Convert flat offsets to 2D coordinates
    row_offsets = offsets // out_w
    col_offsets = offsets % out_w
    
    # Check if we're in the padded region (last row or last column)
    # Original input is in_h x in_w, output is (in_h+1) x (in_w+1)
    is_padded_row = row_offsets >= in_h
    is_padded_col = col_offsets >= in_w
    is_padded = is_padded_row | is_padded_col
    
    # Load scale and bias (they are scalars at position 0)
    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    
    # For non-padded positions, compute the result
    # For padded positions, the result is simply 0 (from constant padding)
    result = tl.zeros_like(offsets, dtype=tl.float32)
    
    # Only compute for valid (non-padded) positions
    valid_mask = ~is_padded & mask
    
    if tl.sum(valid_mask) > 0:
        # Calculate the input indices
        in_row = row_offsets
        in_col = col_offsets
        
        # Calculate flat offset for input
        in_offset = in_row * in_w + in_col
        
        # Load input value
        x = tl.load(in_ptr + in_offset, mask=valid_mask, other=0.0)
        
        # Apply ReLU
        x = tl.maximum(x, 0.0)
        
        # Multiply by scale and add bias
        result = x * scale + bias
        
        # Where mask is false (padded region), result should be 0
        result = tl.where(valid_mask, result, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_relu_mul_add_pad_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper function that launches the Triton kernel.
    in_0: bias (scalar tensor, shape [1])
    in_1: scale (scalar tensor, shape [1])
    in_2: input tensor (shape [B, C, H, W])
    """
    # Get input dimensions
    B, C, in_h, in_w = in_2.shape
    
    # Output dimensions with padding
    out_h = in_h + 1
    out_w = in_w + 1
    
    # Allocate output tensor
    out = torch.empty((B, C, out_h, out_w), device=in_2.device, dtype=in_2.dtype)
    
    # Calculate total elements per channel
    num_elements = out_h * out_w
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel for each channel in the batch
    # We parallelize over the output elements
    grid = (num_programs * B * C,)
    
    # Flatten input and output for kernel
    in_2_flat = in_2.reshape(-1)
    out_flat = out.reshape(-1)
    
    # Launch kernel
    fused_relu_mul_add_pad_kernel[grid](
        in_2_flat,          # in_ptr
        in_1,               # scale_ptr (scalar)
        in_0,               # bias_ptr (scalar)
        out_flat,           # out_ptr
        in_h,               # in_h
        in_w,               # in_w
        out_h,              # out_h
        out_w,              # out_w
        in_w,               # stride_in (for computing flat indices)
        BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_relu_mul_add_pad_kernel_wrapper