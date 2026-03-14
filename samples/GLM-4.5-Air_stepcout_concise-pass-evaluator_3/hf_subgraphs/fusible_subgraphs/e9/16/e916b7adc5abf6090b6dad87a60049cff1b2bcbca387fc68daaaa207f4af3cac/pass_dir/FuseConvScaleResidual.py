import torch
import triton
import triton.language as tl

def pattern(in_8, weight, bias, scale, residual):
    # Match the Conv2D -> Dropout(no-op) -> LayerScale -> Residual pattern
    # Use the exact parameter structure as in the original
    conv_out = torch.conv2d(in_8, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    dropout_out = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    scale_out = dropout_out * scale
    residual_out = residual + scale_out
    return conv_out, residual_out

def replacement_args(in_8, weight, bias, scale, residual):
    return (in_8, weight, bias, scale, residual)

@triton.jit
def fused_conv_scale_residual_kernel(
    input_ptr,      # [N, C_in, H, W] - input tensor
    weight_ptr,     # [C_out, C_in, 1, 1] - conv weight
    bias_ptr,       # [C_out] - conv bias
    scale_ptr,      # [C_out, 1, 1] - layer scale
    residual_ptr,   # [N, C_out, H, W] - residual input
    output_ptr,     # [N, C_out, H, W] - final output
    conv_out_ptr,   # [N, C_out, H, W] - conv output (for intermediate)
    N, C_in, C_out, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate element indices
    pid = tl.program_id(0)
    num_elements = N * C_out * H * W
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Convert linear offsets to tensor indices
    n = offsets // (C_out * H * W)
    remainder = offsets % (C_out * H * W)
    c_out = remainder // (H * W)
    h = (remainder // W) % H
    w = remainder % W
    
    # Load input and weight for conv operation
    # For 1x1 conv, we need one element from input per channel
    input_idx = (n, c_out, h, w)
    residual_idx = (n, c_out, h, w)
    scale_idx = (c_out, 0, 0)
    
    # Load data
    input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    residual_val = tl.load(residual_ptr + residual_idx, mask=mask, other=0.0)
    scale_val = tl.load(scale_ptr + scale_idx, mask=mask, other=0.0)
    
    # Compute 1x1 convolution manually
    conv_val = 0.0
    for c_in in range(C_in):
        weight_idx = (c_out, c_in, 0, 0)
        input_idx_cin = (n, c_in, h, w)
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        input_cin = tl.load(input_ptr + input_idx_cin, mask=mask, other=0.0)
        conv_val += weight_val * input_cin
    
    # Add bias if available
    if bias_ptr is not None:
        bias_idx = (c_out,)
        bias_val = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
        conv_val += bias_val
    
    # Apply layer scaling
    scaled_conv = conv_val * scale_val
    
    # Add residual connection
    final_output = scaled_conv + residual_val
    
    # Store results
    tl.store(conv_out_ptr + input_idx, conv_val, mask=mask)
    tl.store(output_ptr + input_idx, final_output, mask=mask)

@torch.fx.wrap
def fused_conv_scale_residual(input_tensor, weight_tensor, bias_tensor, scale_tensor, residual_tensor):
    N, C_in, H, W = input_tensor.shape
    C_out = weight_tensor.shape[0]
    
    # Determine optimal block size
    total_elements = N * C_out * H * W
    BLOCK_SIZE = 1024  # Can be tuned
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    conv_out = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    output = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    fused_conv_scale_residual_kernel[(num_programs,)](
        input_tensor,
        weight_tensor,
        bias_tensor,
        scale_tensor,
        residual_tensor,
        output,
        conv_out,
        N, C_in, C_out, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return conv_out, output

def replacement_func():
    return fused_conv_scale_residual