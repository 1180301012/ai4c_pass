import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(input_x, conv_weight, running_mean, running_var, bn_weight, bn_bias):
    # Match conv2d + batch_norm + leaky_relu pattern
    conv_out = torch.conv2d(input_x, conv_weight, None, (1, 1), (1, 1), (1, 1), 1)
    bn_out = torch.nn.functional.batch_norm(conv_out, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    leaky_out = torch.nn.functional.leaky_relu(bn_out, 0.01, True)
    return leaky_out

# Argument extraction function
def replacement_args(input_x, conv_weight, running_mean, running_var, bn_weight, bn_bias):
    return (input_x, conv_weight, running_mean, running_var, bn_weight, bn_bias)

@triton.jit
def fused_conv_bn_leaky_kernel(
    x_ptr, 
    weight_ptr,
    running_mean_ptr,
    running_var_ptr, 
    bn_weight_ptr,
    bn_bias_ptr,
    output_ptr,
    N, C_in, H_in, W_in, C_out,
    BLOCK_SIZE_HW: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    # Program IDs for 3D grid: (C_out_block, H_out, W_out)
    c_out_block = tl.program_id(0)
    h_out = tl.program_id(1) 
    w_out = tl.program_id(2)
    
    # Compute output spatial dimensions
    H_out = H_in
    W_out = W_in
    
    # Number of programs per dimension
    num_c_out_blocks = (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    num_h_out = (H_out + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW  
    num_w_out = (W_out + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Return if out of bounds
    if c_out_block >= num_c_out_blocks:
        return
    if h_out >= num_h_out:
        return
    if w_out >= num_w_out:
        return
    
    # Convert to actual spatial coordinates
    h_out = h_out * BLOCK_SIZE_HW
    w_out = w_out * BLOCK_SIZE_HW
    
    # Process a block of output channels
    c_out_start = c_out_block * BLOCK_SIZE_C
    
    # Process each output channel in the block
    for c_idx in range(c_out_start, min(c_out_start + BLOCK_SIZE_C, C_out)):
        # Load batch norm parameters
        running_mean = tl.load(running_mean_ptr + c_idx) if running_mean_ptr else 0.0
        running_var = tl.load(running_var_ptr + c_idx) if running_var_ptr else 1.0
        bn_weight_val = tl.load(bn_weight_ptr + c_idx) if bn_weight_ptr else 1.0
        bn_bias_val = tl.load(bn_bias_ptr + c_idx) if bn_bias_ptr else 0.0
        
        # Initialize output value for this position
        conv_sum = 0.0
        
        # Compute convolution for position (h_out, w_out) for output channel c_idx
        # Iterate over input channels and 3x3 kernel positions
        for c_in in range(C_in):
            # Load all 9 weights for this input/output channel pair at once
            weight_offset = (c_idx * C_in + c_in) * 9  # 3x3 kernel
            kernel_weights = tl.load(weight_ptr + weight_offset + tl.arange(0, 9))
            kernel_weights = kernel_weights.to(tl.float32)
            
            # Process each position in 3x3 kernel with padding
            for kh in range(3):
                for kw in range(3):
                    # Compute input coordinates with padding=1
                    ih = h_out + kh - 1
                    iw = w_out + kw - 1
                    
                    # Check bounds
                    if ih >= 0 and ih < H_in and iw >= 0 and iw < W_in:
                        # Load input value
                        input_offset = (0 * C_in + c_in) * H_in * W_in + ih * W_in + iw
                        x_val = tl.load(x_ptr + input_offset)
                        x_val = x_val.to(tl.float32)
                        
                        # Multiply by corresponding kernel weight
                        weight_val = kernel_weights[kh * 3 + kw]
                        conv_sum += x_val * weight_val
        
        # Batch normalization
        eps = 0.1
        denominator = tl.sqrt(running_var + eps)
        bn_out = (conv_sum - running_mean) / denominator * bn_weight_val + bn_bias_val
        
        # LeakyReLU
        negative_slope = 0.01
        leaky_out = tl.maximum(bn_out, 0.0) + negative_slope * tl.minimum(bn_out, 0.0)
        
        # Store result
        output_offset = c_idx * H_out * W_out + h_out * W_out + w_out
        tl.store(output_ptr + output_offset, leaky_out)

@torch.fx.wrap
def fused_conv_bn_leaky(input_x, conv_weight, running_mean, running_var, bn_weight, bn_bias):
    # Get input tensor dimensions
    batch_size, C_in, H_in, W_in = input_x.shape
    C_out = conv_weight.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, C_out, H_in, W_in), dtype=input_x.dtype, device=input_x.device)
    
    # Move batch norm parameters to input device if they are tensors
    # Use simple isinstance check instead of torch.is_tensor
    if isinstance(running_mean, torch.Tensor):
        running_mean = running_mean.to(input_x.device)
    if isinstance(running_var, torch.Tensor):
        running_var = running_var.to(input_x.device)
    if isinstance(bn_weight, torch.Tensor):
        bn_weight = bn_weight.to(input_x.device)
    if isinstance(bn_bias, torch.Tensor):
        bn_bias = bn_bias.to(input_x.device)
    
    # Set optimal block sizes
    BLOCK_SIZE_HW = 32  # Block size for spatial dimensions
    BLOCK_SIZE_C = 32   # Block size for channels
    
    # Calculate grid dimensions
    num_c_out = (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    num_h_out = (H_in + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    num_w_out = (W_in + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Launch kernel for batch size 1
    fused_conv_bn_leaky_kernel[(num_c_out, num_h_out, num_w_out)](
        input_x,
        conv_weight,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        output,
        batch_size, C_in, H_in, W_in, C_out,
        BLOCK_SIZE_HW, BLOCK_SIZE_C
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_conv_bn_leaky