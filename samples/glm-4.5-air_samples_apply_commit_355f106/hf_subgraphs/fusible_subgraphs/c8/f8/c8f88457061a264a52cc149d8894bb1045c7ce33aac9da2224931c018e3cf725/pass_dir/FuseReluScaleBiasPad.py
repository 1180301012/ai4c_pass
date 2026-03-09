import torch
import triton
import triton.language as tl


# Pattern matching function - matches the computation in model.py
# This includes: relu -> multiply -> add -> pad
# Must return tuple to match model's return structure
def pattern(in_0, in_1, in_2):
    # relu on in_2 (matching model's inplace=False)
    relu_out = torch.nn.functional.relu(in_2, inplace=False)
    # multiply with in_1 (scale)
    mul_out = in_1 * relu_out
    # add in_0 (bias)
    add_out = mul_out + in_0
    # pad with (0, 1, 0, 1)
    pad_out = torch.nn.functional.pad(add_out, (0, 1, 0, 1), 'constant', None)
    return (pad_out,)  # Return tuple to match model's return structure


# Extract arguments for the replacement function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Triton kernel for fused relu + scale + bias + pad
@triton.jit
def fused_relu_scale_bias_pad_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    height,
    width,
    scale_val,
    bias_val,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a 2D block of the output
    # We use program_id(0) for batch*channel dimension and program_id(1) for spatial
    batch_channel = tl.program_id(0)
    spatial = tl.program_id(1)
    
    # Calculate the starting position for this program
    # Each program handles one row of the padded output
    row_offset = spatial
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Input dimensions (before padding)
    input_height = height
    input_width = width
    
    # For columns, we need to handle both original columns and the padded column
    # Original columns: 0 to width-1
    # Padded column: width (index = width)
    
    # scale and bias are already scalar values
    scale = scale_val
    bias = bias_val
    
    # Calculate output values for this row
    # Output shape is (height+1) x (width+1)
    # We process BLOCK_SIZE columns at a time
    
    # Check if we're within bounds
    row_in_bounds = row_offset < input_height + 1
    col_in_bounds = col_offsets < input_width + 1
    
    # For the padded row (row_offset == input_height), all values are 0
    # For the padded column (col_offset == input_width), value is 0
    
    # Determine if this is the padded row
    is_padded_row = row_offset >= input_height
    
    # Calculate input coordinates
    # If row_offset >= input_height, we're in the padded row (value = 0)
    input_row = row_offset
    input_col = col_offsets
    
    # Create mask for valid elements (within original tensor bounds)
    # AND not the padded row AND not the padded column
    is_original_row = row_offset < input_height
    is_original_col = col_offsets < input_width
    is_original = is_original_row and is_original_col
    
    # Load and compute
    # Only load if within bounds and not padded
    input_offset = batch_channel * input_height * input_width + input_row * input_width + input_col
    
    # We need to mask properly
    mask = is_original
    
    # Load from input - clamp indices to valid range for safety
    safe_row = tl.minimum(input_row, input_height - 1)
    safe_col = tl.minimum(input_col, input_width - 1)
    safe_offset = batch_channel * input_height * input_width + safe_row * input_width + safe_col
    
    # Load value and apply ReLU, scale, bias
    x = tl.load(input_ptr + safe_offset, mask=mask, other=0.0)
    
    # ReLU: max(0, x)
    x_relu = tl.where(x > 0, x, 0.0)
    
    # Scale and bias: scale * relu(x) + bias
    result = x_relu * scale + bias
    
    # Store to output (with padding)
    # Output is (height+1) x (width+1)
    output_offset = batch_channel * (input_height + 1) * (input_width + 1) + row_offset * (input_width + 1) + col_offsets
    
    tl.store(output_ptr + output_offset, result, mask=col_in_bounds)


def fused_relu_scale_bias_pad(input_tensor, scale, bias):
    """
    Fused kernel for: relu(input) * scale + bias, with padding (0, 1, 0, 1)
    
    Args:
        input_tensor: Input tensor of shape (B, C, H, W)
        scale: Scalar or tensor of shape (1,)
        bias: Scalar or tensor of shape (1,)
    
    Returns:
        Output tensor of shape (B, C, H+1, W+1)
    """
    B, C, H, W = input_tensor.shape
    
    # Extract scalar values from scale and bias (they have shape [1])
    scale_val = scale.item() if hasattr(scale, 'item') else scale
    bias_val = bias.item() if hasattr(bias, 'item') else bias
    
    # Output shape with padding
    output = torch.empty((B, C, H + 1, W + 1), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Determine block size
    BLOCK_SIZE = 1024
    
    # Grid: (B*C, H+1) - each block handles one row of one channel
    grid = (B * C, H + 1)
    
    # Launch kernel - pass scale and bias as kernel constants
    fused_relu_scale_bias_pad_kernel[grid](
        input_tensor,
        output,
        B * C * H * W,
        H,
        W,
        scale_val,
        bias_val,
        BLOCK_SIZE,
    )
    
    return output


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper function that will be called by the optimized graph.
    in_0: bias (shape [1])
    in_1: scale (shape [1])
    in_2: input tensor
    """
    # Extract scalar values from in_0 and in_1 (they have shape [1])
    bias = in_0.item() if in_0.numel() == 1 else in_0
    scale = in_1.item() if in_1.numel() == 1 else in_1
    
    return fused_relu_scale_bias_pad(in_2, scale, bias)


def replacement_func():
    return kernel_wrapper