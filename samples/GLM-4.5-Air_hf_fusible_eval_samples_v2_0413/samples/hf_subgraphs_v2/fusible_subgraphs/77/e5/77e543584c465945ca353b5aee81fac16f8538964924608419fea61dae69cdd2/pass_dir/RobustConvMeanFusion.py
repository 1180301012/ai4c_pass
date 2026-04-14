import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Pattern that matches the most common conv2d + mean structure
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (1, 1), (1, 1), 256)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def robust_conv_mean_kernel(
    input_ptr,
    weight_ptr,
    conv_out_ptr, 
    mean_out_ptr,
    N, C_in, H, W, H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Safely bound thread indices
    max_channels = tl.minimum(C_out, (pid_c + 1) * BLOCK_SIZE_N)
    
    acc = 0.0
    valid_count = 0
    
    for c in range(pid_c * BLOCK_SIZE_N, max_channels):
        for h in range(H_out):
            for w in range(W_out):
                # Calculate input coordinates
                ih = h * stride_h - pad_h
                iw = w * stride_w - pad_w
                
                # Check bounds before convolution
                if ih >= 0 and iw >= 0 and ih + 2 < H and iw + 2 < W:
                    # Simple 3x3 convolution
                    conv_val = 0.0
                    weight_offset = c * 9
                    
                    for kh in range(3):
                        for kw in range(3):
                            input_h = ih + kh * dilation_h
                            input_w = iw + kw * dilation_w
                            
                            input_offset = (pid_n * C_in * H * W + 
                                          input_h * W + input_w)
                            weight_idx = weight_offset + kh * 3 + kw
                            
                            input_val = tl.load(input_ptr + input_offset).to(tl.float32)
                            weight_val = tl.load(weight_ptr + weight_idx).to(tl.float32)
                            
                            conv_val += input_val * weight_val
                    
                    # Store convolution result
                    output_offset = (pid_n * C_out * H_out * W_out + 
                                   c * H_out * W_out + h * W_out + w)
                    tl.store(conv_out_ptr + output_offset, conv_val.to(input_ptr.type.element_ty))
                    
                    # Accumulate mean computation
                    if pid_n == 0 and h == 0 and w == 0:
                        acc += conv_val
                        valid_count += 1
    
    # Store mean (first thread per batch)
    if pid_c == 0 and pid_n == 0 and valid_count > 0:
        mean_val = acc / valid_count
        mean_offset = pid_n * C_out
        tl.store(mean_out_ptr + mean_offset, mean_val.to(input_ptr.type.element_ty))

@torch.fx.wrap  
def robust_conv_mean_fusion(in_0, in_1):
    N, C_in, H, W = in_1.shape
    C_out = in_0.shape[0]
    
    # Default parameters for common pattern
    stride_h, stride_w = 2, 2
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    groups = 256
    
    # Compute output dimensions
    H_out = (H + 2*pad_h - 3*dilation_h) // stride_h + 1
    W_out = (W + 2*pad_w - 3*dilation_w) // stride_w + 1
    
    # Create output tensors
    conv_out = torch.empty(N, C_out, H_out, W_out, dtype=in_1.dtype, device=in_1.device)
    mean_out = torch.empty(N, C_out, 1, 1, dtype=in_1.dtype, device=in_1.device)
    
    # Set grid dimensions with safe tiling
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = min(64, C_out)
    grid = (N, (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Launch robust kernel
    robust_conv_mean_kernel[grid](
        in_1, in_0, conv_out, mean_out,
        N, C_in, H, W, H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w, 
        dilation_h, dilation_w,
        groups,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Final mean computation
    for n in range(N):
        for c in range(C_out):
            mean_val = conv_out[n, c].mean()
            mean_out[n, c, 0, 0] = mean_val
    
    return conv_out, mean_out

def replacement_func():
    return robust_conv_mean_fusion