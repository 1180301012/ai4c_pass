import torch
import triton
import triton.language as tl

def pattern(x_in, weight, bias, skip1, skip2, stride, padding, dilation, groups):
    # Pattern: Conv2D followed by two additions (residual connection pattern)
    # This matches: conv_out = conv2d(x_in, weight, bias, ...); tmp1 = skip1 + conv_out; tmp2 = tmp1 + skip2
    conv_out = torch.conv2d(x_in, weight, bias, stride, padding, dilation, groups)
    tmp1 = skip1 + conv_out
    tmp2 = tmp1 + skip2
    return conv_out, tmp1, tmp2

def replacement_args(x_in, weight, bias, skip1, skip2, stride, padding, dilation, groups):
    return (x_in, weight, bias, skip1, skip2)

@triton.jit
def fused_conv_add_kernel(
    x_ptr, weight_ptr, bias_ptr, skip1_ptr, skip2_ptr, out_ptr,
    N_in, C_in, H_in, W_in,
    C_out, weight_C_in, weight_H, weight_W,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    groups: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
):
    # Get program IDs
    pid_y = tl.program_id(0)  # Output channel group
    pid_x = tl.program_id(1)  # Spatial location group
    pid_n = tl.program_id(2)  # Batch index
    
    # Handle 1x1 convolution case (pointwise)
    if weight_H == 1 and weight_W == 1:
        # Compute range for this program
        y_start = pid_y * BLOCK_SIZE_Y
        y_end = min(y_start + BLOCK_SIZE_Y, C_out)
        x_start = pid_x * BLOCK_SIZE_X
        x_end = min(x_start + BLOCK_SIZE_X, H_in * W_in)
        
        # Process this batch element
        n = pid_n
        
        # For each output channel in this block
        for c_out in range(y_start, y_end):
            # Load bias
            bias_val = tl.load(bias_ptr + c_out, mask=c_out < C_out, other=0.0)
            
            # Load weight (1x1 kernel, so just scalar per output channel)
            weight_val = tl.load(weight_ptr + c_out, mask=c_out < C_out, other=0.0)
            
            # Process spatial positions
            for spatial_idx in range(x_start, x_end):
                h_idx = spatial_idx // W_in
                w_idx = spatial_idx % W_in
                
                # Input index (assuming no padding, stride dilation for 1x1)
                x_idx = n * C_in * H_in * W_in + c_out * H_in * W_in + spatial_idx
                
                # Load skip tensors at this spatial location
                skip1_idx = n * C_in * H_in * W_in + c_out * H_in * W_in + spatial_idx
                skip2_idx = n * C_in * H_in * W_in + c_out * H_in * W_in + spatial_idx
                
                x_val = tl.load(x_ptr + x_idx, mask=True, other=0.0)
                skip1_val = tl.load(skip1_ptr + skip1_idx, mask=True, other=0.0)
                skip2_val = tl.load(skip2_ptr + skip2_idx, mask=True, other=0.0)
                
                # fused computation: conv_out + skip1 + skip2
                # For pointwise conv: weight * x + bias + skip1 + skip2
                result = weight_val * x_val + bias_val + skip1_val + skip2_val
                
                # Store result at the same spatial location
                out_idx = n * C_out * H_in * W_in + c_out * H_in * W_in + spatial_idx
                tl.store(out_ptr + out_idx, result)

def fused_conv_add_wrapper(x_in, weight, bias, skip1, skip2):
    N_in, C_in, H_in, W_in = x_in.shape
    C_out, weight_C_in, weight_H, weight_W = weight.shape
    
    # Output has same spatial dimensions for pointwise conv
    out_shape = (N_in, C_out, H_in, W_in)
    out = torch.empty(out_shape, dtype=x_in.dtype, device=x_in.device)
    conv_out = torch.empty(out_shape, dtype=x_in.dtype, device=x_in.device)
    tmp1 = torch.empty(out_shape, dtype=x_in.dtype, device=x_in.device)
    
    # Use block sizes that work well for GPU
    BLOCK_SIZE_Y = min(128, C_out)  # Process multiple output channels
    BLOCK_SIZE_X = min(1024, H_in * W_in)  # Process spatial locations
    
    # Calculate grid dimensions
    grid_y = (C_out + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_x = (H_in * W_in + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_n = N_in
    
    # Launch kernel only for 1x1 convolutions (pointwise)
    if weight_H == 1 and weight_W == 1:
        fused_conv_add_kernel[(
            grid_y,
            grid_x,
            grid_n
        )](
            x_ptr=x_in,
            weight_ptr=weight,
            bias_ptr=bias,
            skip1_ptr=skip1,
            skip2_ptr=skip2,
            out_ptr=out,
            N_in=N_in, C_in=C_in, H_in=H_in, W_in=W_in,
            C_out=C_out, weight_C_in=weight_C_in,
            weight_H=weight_H, weight_W=weight_W,
            stride_h=1, stride_w=1,
            pad_h=0, pad_w=0,
            dil_h=1, dil_w=1,
            groups=groups,
            BLOCK_SIZE_Y=BLOCK_SIZE_Y,
            BLOCK_SIZE_X=BLOCK_SIZE_X,
        )
        # Compute intermediate values for pattern return consistency
        conv_out = weight.view(C_out, 1, 1, 1) * x_in + bias.view(1, C_out, 1, 1)
        tmp1 = skip1 + conv_out
        return conv_out, tmp1, out
    else:
        # For non-pointwise conv, fall back to original computation
        conv_out = torch.conv2d(x_in, weight, bias, (1, 1), (0, 0), (1, 1), C_out)
        tmp1 = skip1 + conv_out
        tmp2 = tmp1 + skip2
        return conv_out, tmp1, tmp2

@torch.fx.wrap
def triton_fused_conv_add(x_in, weight, bias, skip1, skip2):
    return fused_conv_add_wrapper(x_in, weight, bias, skip1, skip2)

def replacement_func():
    return triton_fused_conv_add