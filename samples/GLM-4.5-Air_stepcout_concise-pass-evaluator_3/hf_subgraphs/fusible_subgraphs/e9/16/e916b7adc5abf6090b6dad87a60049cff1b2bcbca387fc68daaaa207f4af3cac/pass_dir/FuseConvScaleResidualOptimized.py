import torch
import triton
import triton.language as tl

def pattern(in_8, in_2, in_1, in_0, in_7):
    # Match the exact pattern: Conv2D -> Dropout(no-op) -> LayerScale -> Residual
    tmp_7 = torch.conv2d(in_8, in_2, in_1, (1, 1), (0, 0), (1, 1), 1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    tmp_9 = tmp_8 * in_0
    tmp_10 = in_7 + tmp_9
    return tmp_7, tmp_10

def replacement_args(in_8, in_2, in_1, in_0, in_7):
    return (in_8, in_2, in_1, in_0, in_7)

@triton.jit
def fused_kernel_optimized(
    input_ptr,      # [N, C_in, H, W] - input tensor  
    weight_ptr,     # [C_out, C_in, 1, 1] - conv weights
    bias_ptr,       # [C_out] - conv bias
    scale_ptr,      # [C_out, 1, 1] - layer scale
    residual_ptr,   # [N, C_out, H, W] - residual input
    conv_out_ptr,   # [N, C_out, H, W] - conv output (intermediate)
    output_ptr,     # [N, C_out, H, W] - final output
    N, C_in, C_out, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID and offset calculation  
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
    
    # Load scale parameters (broadcasted)
    scale_idx = (c_out, 0, 0)
    scale_val = tl.load(scale_ptr + scale_idx, mask=mask, other=1.0)
    
    # Load residual input
    residual_idx = (n, c_out, h, w) 
    residual_val = tl.load(residual_ptr + residual_idx, mask=mask, other=0.0)
    
    # Compute 1x1 convolution for each output channel
    conv_val = 0.0
    for c_in in range(C_in):
        # Load weight and input
        weight_idx = (c_out, c_in, 0, 0)
        input_idx = (n, c_in, h, w)
        
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        
        conv_val += weight_val * input_val
    
    # Add bias
    if bias_ptr is not None:
        bias_idx = (c_out,)
        bias_val = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
        conv_val += bias_val
    
    # Apply layer scaling and residual connection
    scaled_conv = conv_val * scale_val
    final_output = scaled_conv + residual_val
    
    # Store results
    conv_out_idx = (n, c_out, h, w)
    output_idx = (n, c_out, h, w)
    
    tl.store(conv_out_ptr + conv_out_idx, conv_val, mask=mask)
    tl.store(output_ptr + output_idx, final_output, mask=mask)

@torch.fx.wrap  
def fused_conv_scale_residual_optimized(input_tensor, weight_tensor, bias_tensor, scale_tensor, residual_tensor):
    N, C_in, H, W = input_tensor.shape
    C_out = weight_tensor.shape[0]
    
    # Use optimal block size for GPU occupancy
    BLOCK_SIZE = 256  # Tune this based on GPU architecture
    total_elements = N * C_out * H * W
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    conv_out = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    output = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set ptrs to None if not available (though they should be in this pattern)
    bias_ptr = bias_tensor if hasattr(bias_tensor, '__cuda_array_interface__') else None
    
    # Launch kernel with optimized grid size
    fused_kernel_optimized[grid_size](
        input_tensor,
        weight_tensor, 
        bias_ptr,
        scale_tensor,
        residual_tensor,
        conv_out,
        output,
        N, C_in, C_out, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return conv_out, output

def replacement_func():
    return fused_conv_scale_residual_optimized