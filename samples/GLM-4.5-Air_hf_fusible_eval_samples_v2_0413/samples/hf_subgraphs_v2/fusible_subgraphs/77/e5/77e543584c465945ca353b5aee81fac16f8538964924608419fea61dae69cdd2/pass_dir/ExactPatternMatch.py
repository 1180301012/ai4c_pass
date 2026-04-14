import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Exact pattern matching stride (1,1) with groups=384
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 384)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def exact_conv_mean_kernel(
    input_ptr,
    weight_ptr, 
    conv_out_ptr,
    mean_out_ptr,
    N, C_out, H, W, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    if pid_h < H_out and pid_w < W_out and pid_c < C_out and pid_n < N:
        # Calculate input coordinates with stride (1,1) and padding (1,1)
        ih = pid_h * 1 - 1  # stride_h=1, pad_h=1
        iw = pid_w * 1 - 1  # stride_w=1, pad_w=1
        
        conv_val = 0.0
        valid_conv = False
        
        # 3x3 convolution with dilation (1,1)
        if ih >= 0 and iw >= 0 and ih + 2 < H and iw + 2 < W:
            valid_conv = True
            # Weight index for this output channel (groups=384 assumed)
            weight_offset = pid_c * 9  # 3x3=9 weights per channel
            
            for kh in range(3):
                for kw in range(3):
                    input_h = ih + kh * 1  # dilation_h=1
                    input_w = iw + kw * 1  # dilation_w=1
                    
                    input_offset = (pid_n * 384 * H * W +  # groups=384
                                  input_h * W + input_w)
                    weight_idx = weight_offset + kh * 3 + kw
                    
                    input_val = tl.load(input_ptr + input_offset).to(tl.float32)
                    weight_val = tl.load(weight_ptr + weight_idx).to(tl.float32)
                    
                    conv_val += input_val * weight_val
        
        # Store convolution result
        output_offset = (pid_n * C_out * H_out * W_out + 
                        pid_c * H_out * W_out + pid_h * W_out + pid_w)
        tl.store(conv_out_ptr + output_offset, conv_val.to(input_ptr.type.element_ty))
        
        # Store mean (simplified - use first valid position per channel)
        if valid_conv and pid_h == 0 and pid_w == 0:
            mean_offset = pid_n * C_out + pid_c
            tl.store(mean_out_ptr + mean_offset, conv_val.to(input_ptr.type.element_ty))

@torch.fx.wrap
def exact_conv_mean(in_0, in_1):
    N, C_in, H, W = in_1.shape
    C_out = in_0.shape[0]
    
    # Parameters for exact match: stride (1,1), padding (1,1), dilation (1,1), groups=384
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    groups = 384
    
    # Compute output dimensions: H_out = (H + 2*pad_h - 3*dilation_h) // stride_h + 1
    H_out = (H + 2*1 - 3*1) // 1 + 1  # Should be H
    W_out = (W + 2*1 - 3*1) // 1 + 1  # Should be W
    
    # Create output tensors
    conv_out = torch.empty(N, C_out, H_out, W_out, dtype=in_1.dtype, device=in_1.device)
    mean_out = torch.empty(N, C_out, 1, 1, dtype=in_1.dtype, device=in_1.device)
    
    # Grid setup
    BLOCK_SIZE = 1
    grid = (N, C_out, H_out, W_out)
    
    # Launch kernel
    exact_conv_mean_kernel[grid](
        in_1, in_0, conv_out, mean_out,
        N, C_out, H, W, H_out, W_out,
        BLOCK_SIZE
    )
    
    # Proper mean computation
    for n in range(N):
        for c in range(C_out):
            mean_val = conv_out[n, c].mean()
            mean_out[n, c, 0, 0] = mean_val
    
    return conv_out, mean_out

def replacement_func():
    return exact_conv_mean