import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Pattern matches conv2d with stride (1,1) and groups=768 followed by mean
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 768)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def conv_mean_kernel(
    input_ptr, weight_ptr, conv_out_ptr, mean_out_ptr,
    N, C_in, H, W, H_out, W_out,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    h_idx = pid_h
    w_idx = pid_w
    
    if h_idx < H_out and w_idx < W_out and pid_c < C_out and pid_n < N:
        ih = h_idx * stride_h - pad_h
        iw = w_idx * stride_w - pad_w
        
        conv_val = 0.0
        valid = False
        
        if ih >= 0 and iw >= 0 and ih + 2 < H and iw + 2 < W:
            valid = True
            weight_offset = pid_c * 9
            
            for kh in range(3):
                for kw in range(3):
                    input_h = ih + kh * dilation_h
                    input_w = iw + kw * dilation_w
                    
                    input_offset = pid_n * C_in * H * W + input_h * W + input_w
                    weight_idx = weight_offset + kh * 3 + kw
                    
                    input_val = tl.load(input_ptr + input_offset).to(tl.float32)
                    weight_val = tl.load(weight_ptr + weight_idx).to(tl.float32)
                    
                    conv_val += input_val * weight_val
        
        output_offset = (pid_n * C_out * H_out * W_out + 
                        pid_c * H_out * W_out + h_idx * W_out + w_idx)
        tl.store(conv_out_ptr + output_offset, conv_val.to(input_ptr.type.element_ty))
        
        if valid and pid_h == 0 and pid_w == 0:
            mean_offset = pid_n * C_out + pid_c
            tl.store(mean_out_ptr + mean_offset, conv_val.to(input_ptr.type.element_ty))

@torch.fx.wrap
def conv_mean_fusion(in_0, in_1):
    N, C_in, H, W = in_1.shape
    C_out = in_0.shape[0]
    
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    groups = 768
    
    H_out = (H + 2*pad_h - 3*dilation_h) // stride_h + 1
    W_out = (W + 2*pad_w - 3*dilation_w) // stride_w + 1
    
    conv_out = torch.empty(N, C_out, H_out, W_out, dtype=in_1.dtype, device=in_1.device)
    mean_out = torch.empty(N, C_out, 1, 1, dtype=in_1.dtype, device=in_1.device)
    
    grid = (N, C_out, H_out, W_out)
    
    conv_mean_kernel[grid](
        in_1, in_0, conv_out, mean_out,
        N, C_in, H, W, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
        1
    )
    
    for n in range(N):
        for c in range(C_out):
            mean_val = conv_out[n, c, 0, 0]
            mean_out[n, c, 0, 0] = mean_val
    
    return conv_out, mean_out

def replacement_func():
    return conv_mean_fusion