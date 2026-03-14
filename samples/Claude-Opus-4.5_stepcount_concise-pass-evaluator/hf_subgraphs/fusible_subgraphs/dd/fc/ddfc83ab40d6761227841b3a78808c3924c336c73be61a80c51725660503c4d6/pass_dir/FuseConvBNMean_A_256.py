import torch
import triton
import triton.language as tl

# Pattern A: conv2d(in_6) + in_7 + in_6 -> batch_norm -> mean, groups=256
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    tmp_6 = torch.conv2d(in_6, in_5, in_4, (1, 1), (0, 0), (1, 1), 256)
    tmp_7 = in_7 + tmp_6
    tmp_8 = tmp_7 + in_6
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_10 = tmp_9.mean((2, 3), keepdim=True)
    return tmp_9, tmp_10

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)

@triton.jit
def fused_kernel(
    x_ptr, y_ptr, conv_w_ptr, conv_b_ptr,
    bn_mean_ptr, bn_var_ptr, bn_gamma_ptr, bn_beta_ptr,
    out_ptr, mean_ptr,
    N, C, HW,
    stride_n, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    w = tl.load(conv_w_ptr + c)
    b = tl.load(conv_b_ptr + c)
    mean = tl.load(bn_mean_ptr + c)
    var = tl.load(bn_var_ptr + c)
    gamma = tl.load(bn_gamma_ptr + c)
    beta = tl.load(bn_beta_ptr + c)
    
    inv_std = tl.rsqrt(var + 1e-5)
    bn_scale = gamma * inv_std
    bn_shift = beta - gamma * mean * inv_std
    w_plus_1 = w + 1.0
    
    final_scale_x = bn_scale * w_plus_1
    final_scale_y = bn_scale
    final_bias = bn_scale * b + bn_shift
    
    base_offset = n * stride_n + c * stride_c
    
    sum_val = 0.0
    
    num_iters = tl.cdiv(HW, BLOCK_SIZE)
    for i in range(num_iters):
        block_start = i * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        
        x = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + base_offset + offsets, mask=mask, other=0.0)
        
        out = x * final_scale_x + y * final_scale_y + final_bias
        
        tl.store(out_ptr + base_offset + offsets, out, mask=mask)
        sum_val += tl.sum(tl.where(mask, out, 0.0))
    
    mean_offset = n * C + c
    tl.store(mean_ptr + mean_offset, sum_val / HW)

@torch.fx.wrap
def fused_kernel_impl(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    N, C, H, W = in_6.shape
    HW = H * W
    
    out = torch.empty_like(in_6)
    mean_out = torch.empty((N, C, 1, 1), dtype=in_6.dtype, device=in_6.device)
    
    conv_w = in_5.view(C)
    
    stride_n = C * HW
    stride_c = HW
    
    BLOCK_SIZE = 256
    grid = (N * C,)
    fused_kernel[grid](
        in_6, in_7, conv_w, in_4,
        in_0, in_1, in_3, in_2,
        out, mean_out.view(N, C),
        N, C, HW,
        stride_n, stride_c,
        BLOCK_SIZE,
    )
    
    return out, mean_out

def fused_conv_bn_mean(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    out, mean_out = fused_kernel_impl(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)
    return out, mean_out

def replacement_func():
    return fused_conv_bn_mean