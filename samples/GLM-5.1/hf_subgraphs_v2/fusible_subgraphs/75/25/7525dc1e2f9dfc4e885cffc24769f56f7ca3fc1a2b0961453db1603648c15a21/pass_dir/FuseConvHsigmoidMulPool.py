import torch
import triton
import triton.language as tl

# Pattern matching function - must mirror model.py exactly
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardsigmoid(tmp_2, False)
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return (tmp_7,)

# Argument extraction - pass inputs needed for the replacement
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_pool_mean_kernel(
    in_2_ptr,
    out_mean_ptr,
    B, C_in, H, W, HW,
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_HW: tl.constexpr,
):
    """
    Compute spatial mean of in_2 for each (b, c) pair.
    in_2 is [B, C_in, H, W] - average over spatial dims for each (b, c).
    Output: out_mean [B, C_in] = mean of in_2 over spatial dims.
    
    Algebraic simplification:
    avg_pool(hardsigmoid(conv) * in_2) = hardsigmoid(conv) * avg_pool(in_2)
    """
    pid = tl.program_id(0)  # linear index over B*C_in
    
    b = pid // C_in
    c = pid % C_in
    
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    
    for hw_start in range(0, HW, BLOCK_HW):
        hw_off = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_off < HW
        
        h_coord = hw_off // W
        w_coord = hw_off % W
        
        ptr = in_2_ptr + b * stride_b + c * stride_c + h_coord * stride_h + w_coord * stride_w
        vals = tl.load(ptr, mask=hw_mask, other=0.0)
        acc += vals
    
    mean_val = tl.sum(acc) / HW
    tl.store(out_mean_ptr + pid, mean_val)


@triton.jit
def fused_conv_hsigmoid_mul_kernel(
    in_3_ptr,
    weight_ptr,
    bias_ptr,
    mean_in2_ptr,
    out_ptr,
    C_in, C_out,
    stride_in3_b, stride_in3_c, stride_in3_h, stride_in3_w,
    stride_w_oc, stride_w_ic, stride_w_h, stride_w_w,
    stride_mean_b, stride_mean_c,
    stride_out_b, stride_out_c,
    BLOCK_CIN: tl.constexpr,
):
    """
    For each output element (b, c_out):
    1. conv1x1: out = in_3[b,:] @ weight[c_out,:] + bias[c_out]
       Since in_3 is [B, C_in, 1, 1], this is just a dot product.
    2. hardsigmoid: clamp(x + 3, 0, 6) / 6
    3. multiply by mean_in2[b, c_out]
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    bias_val = tl.load(bias_ptr + pid_c).to(tl.float32)
    
    acc = tl.zeros([BLOCK_CIN], dtype=tl.float32)
    
    for c_in_start in range(0, C_in, BLOCK_CIN):
        c_in_off = c_in_start + tl.arange(0, BLOCK_CIN)
        c_in_mask = c_in_off < C_in
        
        in3_vals = tl.load(
            in_3_ptr + pid_b * stride_in3_b + c_in_off * stride_in3_c + 0 * stride_in3_h + 0 * stride_in3_w,
            mask=c_in_mask, other=0.0
        ).to(tl.float32)
        
        w_vals = tl.load(
            weight_ptr + pid_c * stride_w_oc + c_in_off * stride_w_ic + 0 * stride_w_h + 0 * stride_w_w,
            mask=c_in_mask, other=0.0
        ).to(tl.float32)
        
        acc += in3_vals * w_vals
    
    conv_out = tl.sum(acc) + bias_val
    
    # hardsigmoid: clamp(x + 3, 0, 6) / 6
    val = conv_out + 3.0
    hsigmoid = tl.minimum(tl.maximum(val, 0.0), 6.0) / 6.0
    
    mean_val = tl.load(mean_in2_ptr + pid_b * stride_mean_b + pid_c * stride_mean_c).to(tl.float32)
    
    result = hsigmoid * mean_val
    tl.store(out_ptr + pid_b * stride_out_b + pid_c * stride_out_c, result)


@torch.fx.wrap
def fused_se_attn(in_0, in_1, in_2, in_3):
    """
    Fused implementation of the SE attention block.
    
    Algebraic simplification:
    avg_pool(hardsigmoid(conv) * in_2) = hardsigmoid(conv) * avg_pool(in_2)
    since hardsigmoid(conv) is constant across spatial dims (1x1 → broadcast).
    
    This eliminates the large [B, C, H, W] intermediate tensor!
    """
    B = in_2.shape[0]
    C_in = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    C_out = in_1.shape[0]
    HW = H * W
    
    # Step 1: Compute spatial mean of in_2
    mean_in2 = torch.empty(B, C_in, dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_HW = 128 if HW >= 128 else 64
    
    fused_pool_mean_kernel[(B * C_in,)](
        in_2, mean_in2,
        B, C_in, H, W, HW,
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        BLOCK_HW=BLOCK_HW,
    )
    
    # Step 2: conv1x1 + hardsigmoid + multiply with mean
    out = torch.empty(B, C_out, dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_CIN = 64
    
    fused_conv_hsigmoid_mul_kernel[(B, C_out)](
        in_3, in_1, in_0, mean_in2, out,
        C_in, C_out,
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        mean_in2.stride(0), mean_in2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_CIN=BLOCK_CIN,
    )
    
    return (out,)


def replacement_func():
    return fused_se_attn