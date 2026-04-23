import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_cat_bn_prelu_kernel(
    x_ptr, conv_weight_ptr, in_6_ptr, bn_mean_ptr, bn_var_ptr, 
    bn_weight_ptr, bn_bias_ptr, prelu_weight_ptr,
    out_ptr,
    N, C_in, H, W,
    C_conv_out: tl.constexpr,
    bn_eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: process each element in the output
    pid = tl.program_id(0)
    n_elements = N * (C_conv_out * 2) * H * W
    if pid >= n_elements:
        return
    
    # Compute output coordinates
    temp = pid
    n = temp // ((C_conv_out * 2) * H * W)
    temp = temp % ((C_conv_out * 2) * H * W)
    c = temp // (H * W)
    temp = temp % (H * W)
    h = temp // W
    w = temp % W
    
    # Load prelu weight (channel-wise)
    prelu_w = tl.load(prelu_weight_ptr + c)
    
    # Convolution parameters
    stride = 1
    padding = 4
    dilation = 4
    
    if c < C_conv_out:
        # Second branch: depthwise convolution on x
        conv_out = 0.0
        for kh in range(0, 3):
            for kw in range(0, 3):
                # For dilated conv with stride=1, padding=4, dilation=4:
                # h_out = h_in * dilation + kh - padding
                # So h_in = (h_out + padding - kh) // dilation
                h_in = (h + padding - kh) // dilation
                w_in = (w + padding - kw) // dilation
                
                # Check if within valid input range (before padding)
                if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                    x_idx = ((n * C_in + c) * H + h_in) * W + w_in
                    x_val = tl.load(x_ptr + x_idx)
                    
                    # Load conv weight for this channel and kernel position
                    weight_idx = c * 9 + kh * 3 + kw
                    w_val = tl.load(conv_weight_ptr + weight_idx)
                    
                    conv_out += x_val * w_val
        
        # Apply batch norm (second branch)
        bn_mean_val = tl.load(bn_mean_ptr + c)
        bn_var_val = tl.load(bn_var_ptr + c)
        bn_weight_val = tl.load(bn_weight_ptr + c)
        bn_bias_val = tl.load(bn_bias_ptr + c)
        
        bn_std = tl.sqrt(bn_var_val + bn_eps)
        bn_norm = (conv_out - bn_mean_val) / bn_std * bn_weight_val + bn_bias_val
        
        # Apply PReLU
        out_val = tl.where(bn_norm < 0, bn_norm * prelu_w, bn_norm)
        
        # Store main output
        out_idx = ((n * (C_conv_out * 2) + c) * H + h) * W + w
        tl.store(out_ptr + out_idx, out_val)
    else:
        # First branch: passthrough in_6
        c_first = c - C_conv_out
        out_val = tl.load(in_6_ptr + ((n * C_conv_out + c_first) * H + h) * W + w)
        
        # Apply batch norm (first branch)
        bn_mean_val = tl.load(bn_mean_ptr + c)
        bn_var_val = tl.load(bn_var_ptr + c)
        bn_weight_val = tl.load(bn_weight_ptr + c)
        bn_bias_val = tl.load(bn_bias_ptr + c)
        
        bn_std = tl.sqrt(bn_var_val + bn_eps)
        bn_norm = (out_val - bn_mean_val) / bn_std * bn_weight_val + bn_bias_val
        
        # Apply PReLU
        out_val = tl.where(bn_norm < 0, bn_norm * prelu_w, bn_norm)
        
        # Store main output
        out_idx = ((n * (C_conv_out * 2) + c) * H + h) * W + w
        tl.store(out_ptr + out_idx, out_val)


@triton.jit
def compute_avg_pool_view_kernel(
    out_ptr, out_view_ptr,
    N, C_total, H, W,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each program handles one channel for a single sample
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    if pid_n >= N or pid_c >= C_total:
        return
    
    sum_val = 0.0
    for h in range(0, H):
        for w in range(0, W):
            idx = ((pid_n * C_total + pid_c) * H + h) * W + w
            sum_val += tl.load(out_ptr + idx)
    
    avg_val = sum_val / (H * W)
    # Store in view output format
    view_idx = pid_n * C_total + pid_c
    tl.store(out_view_ptr + view_idx, avg_val)


@torch.fx.wrap
def fused_conv_cat_bn_prelu(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Fused kernel: conv2d + cat + batch_norm + prelu + adaptive_avg_pool2d + view
    in_0: prelu weight [128]
    in_1: bn running mean [128]
    in_2: bn running var [128]
    in_3: bn bias [128]
    in_4: bn weight [128]
    in_5: conv weight [64, 1, 3, 3]
    in_6: passthrough tensor [N, 64, H, W] 
    in_7: conv input tensor [N, 64, H, W]
    Returns: (main_output [N, 128, H, W], pooled_view [N, 128])
    """
    N, C_in, H, W = in_7.shape
    C_conv_out = 64
    C_total = 128
    bn_eps = 0.001
    
    # Output tensor after conv + cat + bn + prelu
    out = torch.empty((N, C_total, H, W), dtype=in_7.dtype, device=in_7.device)
    
    # Grid dimensions
    n_elements = N * C_total * H * W
    BLOCK_SIZE = 128
    
    # Launch conv+cat+bn+prelu kernel
    grid_conv = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_conv_cat_bn_prelu_kernel[grid_conv](
        in_7, in_5, in_6, in_1, in_2, in_4, in_3, in_0,
        out,
        N, C_in, H, W,
        C_conv_out,
        bn_eps,
        BLOCK_SIZE,
    )
    
    # Adaptive average pool + view
    out_view = torch.empty((N, C_total), dtype=in_7.dtype, device=in_7.device)
    
    grid_pool = (N, C_total)
    compute_avg_pool_view_kernel[grid_pool](
        out, out_view,
        N, C_total, H, W,
        1,  # BLOCK_SIZE_H
        1,  # BLOCK_SIZE_W
    )
    
    return out, out_view


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Match the pattern: conv2d + cat + batch_norm + prelu + adaptive_avg_pool2d + view
    Returns the main output (after prelu) and the pooled view output
    """
    tmp_6 = torch.conv2d(in_7, in_5, None, (1, 1), (4, 4), (4, 4), 64)
    tmp_7 = torch.cat([in_6, tmp_6], 1)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    tmp_9 = torch.prelu(tmp_8, in_0)
    tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1)
    tmp_11 = tmp_10.view(tmp_10.shape[0], 128)
    return tmp_9, tmp_11


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)


def replacement_func():
    return fused_conv_cat_bn_prelu