import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """
    Match the Conv2D + ReLU + Dropout pattern from model.py
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(tmp_2, inplace=True)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized fused kernel
@triton.jit
def fused_conv2d_relu_kernel(
    x_ptr,  # input [N, C_in, H_in, W_in]
    weight_ptr,  # weight [C_out, C_in, kH, kW]
    bias_ptr,  # bias [C_out]
    out_ptr,  # output [N, C_out, H_out, W_out]
    N, C_in, H_in, W_in, C_out,
    H_out, W_out,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Grid setup
    batch_idx = tl.program_id(0)
    out_c = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)
    
    # Calculate output position
    out_offset = batch_idx * C_out * H_out * W_out + out_c * H_out * W_out + h_out * W_out + w_out
    out_ptr_base = out_ptr + out_offset
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + out_c)
    
    # Accumulator
    acc = bias_val
    
    # Loop over input channels
    for c_in in range(0, C_in, BLOCK_C):
        c_in_end = min(c_in + BLOCK_C, C_in)
        
        # Load weight block
        weight_offset = out_c * C_in * 7 * 7 + c_in * 7 * 7
        weight_ptr_block = weight_ptr + weight_offset
        
        # Load input block for this batch and input channel range
        x_ptr_batch_base = x_ptr + batch_idx * C_in * H_in * W_in + c_in * H_in * W_in
        
        # Calculate input spatial position for this output position
        input_h = h_out * stride_h - padding_h
        input_w = w_out * stride_w - padding_w
        
        # Handle convolution with dilation
        for kh in range(7):
            for kw in range(7):
                # Calculate input position with dilation
                dilated_h = input_h + kh * dilation_h
                dilated_w = input_w + kw * dilation_w
                
                # Check bounds
                if 0 <= dilated_h < H_in and 0 <= dilated_w < W_in:
                    # Load weight
                    weight_idx = (kh * 7 + kw)
                    weight_val = tl.load(weight_ptr_block + weight_idx, other=0.0)
                    
                    # Load input
                    x_idx = dilated_h * W_in + dilated_w
                    x_val = tl.load(x_ptr_batch_base + x_idx, other=0.0)
                    
                    # Multiply accumulate
                    acc += weight_val * x_val
                # else: implicitly zero (handled by other=0.0)
    
    # Apply ReLU activation
    out_val = tl.math.maximum(acc, 0.0)
    
    # Apply dropout with p=0.0 (identity operation)
    out_val_final = out_val
    
    # Store result
    tl.store(out_ptr_base, out_val_final)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv2d_relu(in_0, in_1, in_2):
    # Get input shapes
    N, C_in, H_in, W_in = in_2.shape
    C_out = in_1.shape[0]
    
    # Calculate output dimensions
    H_out = (H_in + 2*0 - 1*(7-1) - 1) // 1 + 1
    W_out = (W_in + 2*0 - 1*(7-1) - 1) // 1 + 1
    
    # Create output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=in_2.dtype, device=in_2.device)
    
    # Grid configuration
    grid = (N, C_out, H_out, W_out)
    
    # Launch kernel
    fused_conv2d_relu_kernel[grid](
        in_2,
        in_1,
        in_0,
        out,
        N, C_in, H_in, W_in, C_out,
        H_out, W_out,
        1, 1,  # stride
        0, 0,  # padding
        1, 1,  # dilation
        1,     # groups
        32, 32  # block sizes
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv2d_relu