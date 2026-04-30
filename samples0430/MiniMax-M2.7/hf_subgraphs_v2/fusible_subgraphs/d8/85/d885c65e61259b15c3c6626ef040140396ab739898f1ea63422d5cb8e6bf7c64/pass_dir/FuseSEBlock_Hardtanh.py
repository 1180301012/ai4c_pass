import torch
import triton
import triton.language as tl

# Simple but effective kernel for SE block fusion
@triton.jit
def fused_se_block_kernel(
    x_ptr, se_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C_out, H, W, C_in,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    stride_se_b, stride_se_c,
    stride_wc, stride_wk
):
    # Get program ID - each program handles one channel in one batch
    pid = tl.program_id(0)
    num_programs = B * C_out
    
    if pid >= num_programs:
        return
    
    # Compute batch and channel indices
    b = pid // C_out
    c = pid % C_out
    
    # Compute base offsets for this batch/channel
    x_offset = b * stride_xb + c * stride_xc
    out_offset = b * stride_ob + c * stride_oc
    se_offset = b * stride_se_b
    weight_offset = c * stride_wc
    
    # Load bias
    bias_val = tl.load(bias_ptr + c)
    
    # Compute SE block output (1x1 conv)
    conv_out = bias_val
    for k in range(C_in):
        se_val = tl.load(se_ptr + se_offset + k * stride_se_c)
        weight_val = tl.load(weight_ptr + weight_offset + k * stride_wk)
        conv_out = conv_out + se_val * weight_val
    
    # Compute sigmoid - cast to float32 for exp compatibility, then back
    neg_exp = -conv_out
    exp_neg = tl.exp(neg_exp.to(tl.float32)).to(conv_out.dtype)
    sigmoid_out = 1.0 / (1.0 + exp_neg)
    
    # Process all H*W spatial positions
    total_elements = H * W
    for spatial_idx in range(total_elements):
        h = spatial_idx // W
        w = spatial_idx % W
        
        # Load x value
        x_idx = x_offset + h * stride_xh + w * stride_xw
        x_val = tl.load(x_ptr + x_idx)
        
        # Multiply and apply hardtanh
        mul_val = x_val * sigmoid_out
        out_val = tl.minimum(tl.maximum(mul_val, 0.0), 6.0)
        
        # Store result
        out_idx = out_offset + h * stride_oh + w * stride_ow
        tl.store(out_ptr + out_idx, out_val)


@torch.fx.wrap
def fused_se_block_wrapper(bias, weight, x, se):
    """
    Fused SE block kernel: conv2d + sigmoid + multiply + hardtanh
    Args: bias=in_0 (bias tensor), weight=in_1 (weight tensor), x=in_2 (feature tensor), se=in_3 (SE tensor)
    """
    # Get dimensions using tensor introspection
    bias_shape = list(bias.shape) if hasattr(bias.shape, '__iter__') else [bias.numel()]
    weight_shape = list(weight.shape) if hasattr(weight.shape, '__iter__') else [weight.numel()]
    x_shape = list(x.shape) if hasattr(x.shape, '__iter__') else [x.numel()]
    se_shape = list(se.shape) if hasattr(se.shape, '__iter__') else [se.numel()]
    
    # Handle framework wrapping by getting total elements
    bias_numel = bias.numel()
    weight_numel = weight.numel()
    x_numel = x.numel()
    se_numel = se.numel()
    
    # Get C_out and C_in from tensor dimensions
    # bias: [C_out], weight: [C_out, C_in, 1, 1]
    C_out = bias_shape[0] if len(bias_shape) >= 1 else bias_numel
    C_in = weight_shape[1] if len(weight_shape) >= 2 else (weight_numel // C_out)
    
    # Try to determine B from se tensor
    # se tensor: [B, C_in, 1, 1]
    if len(se_shape) >= 2:
        B = se_shape[0]
    else:
        B = se_numel // (C_in * 1 * 1) if se_numel > C_in else 1
    B = max(1, B)
    
    # x tensor: [B, C_out, H, W]
    # We know x_numel = B * C_out * H * W
    # So H * W = x_numel / (B * C_out)
    HW = x_numel // (B * C_out) if B * C_out > 0 else x_numel
    H = int(HW ** 0.5) if HW > 0 else 1
    W = HW // H if H > 0 else 1
    
    # Handle edge case where sqrt gives non-integer
    if H * W != HW:
        # Try different factorizations
        for h_test in range(1, int(HW ** 0.5) + 2):
            if HW % h_test == 0:
                H = h_test
                W = HW // h_test
                break
    
    # Ensure we have reasonable values
    H = max(1, H)
    W = max(1, W)
    
    # Allocate output tensor using allowed API
    out = torch.empty((B, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Compute strides for 4D tensors
    stride_xb = C_out * H * W
    stride_xc = H * W
    stride_xh = W
    stride_xw = 1
    
    stride_ob = C_out * H * W
    stride_oc = H * W
    stride_oh = W
    stride_ow = 1
    
    stride_se_b = C_in * 1 * 1
    stride_se_c = 1 * 1
    
    stride_wc = C_in * 1 * 1
    stride_wk = 1
    
    # Launch kernel - one program per (batch, channel)
    num_programs = B * C_out
    
    fused_se_block_kernel[(num_programs,)](
        x, se, weight, bias, out,
        B, C_out, H, W, C_in,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_ob, stride_oc, stride_oh, stride_ow,
        stride_se_b, stride_se_c,
        stride_wc, stride_wk
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the SE block pattern:
    conv2d(in_3, in_1, in_0) -> sigmoid -> multiply with in_2 -> hardtanh
    """
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_se_block_wrapper