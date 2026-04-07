import torch
import triton
import triton.language as tl

def pattern(pixel_values, weight_tensor, bias_tensor):
    """
    Pattern that matches: conv3d + flatten + transpose operations.
    This fuses conv3d computation with reshape operations for better performance.
    """
    conv3d = torch.conv3d(pixel_values, weight_tensor, bias_tensor, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_4 = conv3d.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    return tmp_5

def replacement_args(pixel_values, weight_tensor, bias_tensor):
    return (pixel_values, weight_tensor, bias_tensor)

@triton.jit
def fused_conv3d_flatten_transpose_kernel(
    input_ptr,      # [B, C_in, T_in, H_in, W_in]
    weight_ptr,     # [C_out, C_in, K_t, K_h, K_w]
    bias_ptr,       # [C_out]
    output_ptr,     # [B, T_out*H_out*W_out, C_out]
    batch_size,
    in_channels,
    in_time, in_height, in_width,
    out_channels,
    k_time, k_height, k_width,
    stride_t, stride_h, stride_w,
    pad_t, pad_h, pad_w,
):
    """
    Optimized kernel that fuses conv3d + flatten + transpose operations.
    Using a simplified approach to avoid compilation errors.
    """
    # Program identifier
    pid = tl.program_id(0)
    batch_offset = pid * in_channels * in_time * in_height * in_width
    
    # Early exit if batch index is out of bounds
    if pid >= batch_size:
        return
    
    # Calculate output spatial dimensions
    out_time = (in_time - k_time) // stride_t + 1
    out_height = (in_height - k_height) // stride_h + 1
    out_width = (in_width - k_width) // stride_w + 1
    total_spatial = out_time * out_height * out_width
    
    # Process each output spatial position
    for sp_idx in range(total_spatial):
        # Convert 1D spatial index to 3D coordinates
        t_out = sp_idx // (out_height * out_width)
        remainder = sp_idx % (out_height * out_width)
        h_out = remainder // out_width
        w_out = remainder % out_width
        
        # Compute each output channel
        offset = pid * total_spatial * out_channels + sp_idx * out_channels
        
        for c_out in range(out_channels):
            # Initialize accumulator with bias
            acc = tl.load(bias_ptr + c_out).to(tl.float32)
            
            # Convolution computation
            for c_in in range(in_channels):
                t_start = t_out * stride_t
                h_start = h_out * stride_h
                w_start = w_out * stride_w
                
                for kt in range(k_time):
                    for kh in range(k_height):
                        for kw in range(k_width):
                            t_in = t_start + kt
                            h_in = h_start + kh
                            w_in = w_start + kw
                            
                            # Check bounds and load input/weights if within bounds
                            t_valid = t_in < in_time
                            h_valid = h_in < in_height
                            w_valid = w_in < in_width
                            
                            valid = t_valid and h_valid and w_valid
                            
                            if valid:
                                # Load input and weights
                                input_idx = batch_offset + \
                                           c_in * (in_time * in_height * in_width) + \
                                           t_in * (in_height * in_width) + \
                                           h_in * in_width + w_in
                                weight_idx = c_out * (in_channels * k_time * k_height * k_width) + \
                                           c_in * (k_time * k_height * k_width) + \
                                           kt * (k_height * k_width) + \
                                           kh * k_width + kw
                                
                                input_val = tl.load(input_ptr + input_idx).to(tl.float32)
                                weight_val = tl.load(weight_ptr + weight_idx).to(tl.float32)
                                
                                # Accumulate
                                acc += input_val * weight_val
            
            # Store result
            output_idx = offset + c_out
            tl.store(output_ptr + output_idx, acc)

@torch.fx.wrap
def fused_conv3d_flatten_transpose(input_tensor, weight_tensor, bias_tensor):
    """
    GPU kernel that fuses conv3d + flatten + transpose operations.
    Directly produces output in transformer sequence format [B, T*H*W, C].
    """
    # Get input dimensions
    batch_size, in_channels, in_time, in_height, in_width = input_tensor.shape
    out_channels, _, k_time, k_height, k_width = weight_tensor.shape
    
    # Calculate output dimensions
    stride_t, stride_h, stride_w = 2, 16, 16  # Fixed from pattern
    out_time = (in_time - k_time) // stride_t + 1
    out_height = (in_height - k_height) // stride_h + 1
    out_width = (in_width - k_width) // stride_w + 1
    total_spatial = out_time * out_height * out_width
    
    # Output shape: [batch_size, total_spatial, out_channels]
    # Store as [B, T*H*W, C] directly (sequence format)
    output_shape = (batch_size, total_spatial, out_channels)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel - one program per batch
    grid = batch_size
    
    fused_conv3d_flatten_transpose_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_time=in_time,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        k_time=k_time,
        k_height=k_height,
        k_width=k_width,
        stride_t=stride_t,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_t=0,
        pad_h=0,
        pad_w=0,
    )
    
    return output

def replacement_func():
    return fused_conv3d_flatten_transpose