import torch
import triton
import triton.language as tl
import math

def pattern(in_3, in_1, in_0):
    conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(conv2d, inplace=True)
    return conv2d, tmp_3

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def simple_fused_conv2d_relu_kernel(
    x_ptr, w_ptr, b_ptr,
    output_ptr,
    N, H_in, W_in, C_in,
    C_out, K,
    stride_h, stride_w,
    pad_h, pad_w,
    H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    output_size = H_out * W_out * C_out
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, output_size)
    
    for idx in range(start_idx, end_idx):
        h_out = idx // (W_out * C_out)
        w_out = (idx % (W_out * C_out)) // C_out
        c_out = idx % C_out
        
        val = 0.0
        
        for kh in range(K):
            for kw in range(K):
                h_in = h_out * stride_h - pad_h + kh
                w_in = w_out * stride_w - pad_w + kw
                
                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                    offset_in = (h_in * W_in + w_in) * C_in
                    offset_weight = (kh * K + kw) * C_in
                    offset_out = h_in * W_in + w_in
                    
                    for ci in range(C_in):
                        x_val = tl.load(x_ptr + N * (offset_in + ci), mask=None, other=0.0)
                        w_val = tl.load(w_ptr + C_out * (offset_weight + ci), mask=None, other=0.0)
                        val += x_val * w_val
        
        bias_val = tl.load(b_ptr + c_out, mask=None, other=0.0)
        val += bias_val
        
        val = tl.maximum(val, 0.0)
        
        tl.store(output_ptr + idx, val)

@torch.fx.wrap
def fused_conv2d_relu(in_3, in_1, in_0):
    N, C_in, H_in, W_in = in_3.shape
    C_out, K, _, _ = in_1.shape
    
    H_out = (H_in + 2 * 1 - K) // 2 + 1
    W_out = (W_in + 2 * 1 - K) // 2 + 1
    
    output = torch.zeros((N, C_out, H_out, W_out), dtype=in_3.dtype, device=in_3.device)
    
    BLOCK_SIZE = 1024
    num_programs = (N * H_out * W_out * C_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_fused_conv2d_relu_kernel[(num_programs,)](
        in_3, in_1, in_0,
        output,
        N, H_in, W_in, C_in,
        C_out, K,
        2, 2,
        1, 1,
        H_out, W_out,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_conv2d_relu