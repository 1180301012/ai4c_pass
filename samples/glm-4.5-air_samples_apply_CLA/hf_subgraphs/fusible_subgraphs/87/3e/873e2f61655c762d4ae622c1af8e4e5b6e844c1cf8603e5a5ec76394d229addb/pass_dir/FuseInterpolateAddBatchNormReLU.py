import torch
import triton
import triton.language as tl

# Pattern: interpolate + add + batch_norm + relu
# Input: (N, C, H, W) -> interpolate to (N, C, 64, 64) -> add -> bn -> relu
# Only return the final value that is used externally
def pattern(tmp_6, in_7, mean, var, weight, bias):
    tmp_7 = torch.nn.functional.interpolate(tmp_6, (64, 64), None, 'bilinear', False)
    tmp_8 = in_7 + tmp_7
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, mean, var, weight, bias, False, 0.1, 1e-05)
    tmp_10 = torch.nn.functional.relu(tmp_9, inplace=True)
    return tmp_10

def replacement_args(tmp_6, in_7, mean, var, weight, bias):
    return (tmp_6, in_7, mean, var, weight, bias)


# Optimized Triton kernel that fuses interpolate + add + batch_norm + relu
@triton.jit
def fused_interpolate_add_bn_relu_kernel(
    input_ptr, other_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    orig_H: tl.constexpr, orig_W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one channel
    pid = tl.program_id(0)
    if pid >= N * C:
        return
    
    n = pid // C
    c = pid % C
    
    # Load batch norm parameters for this channel
    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)
    
    # Compute normalized standard deviation
    std = tl.sqrt(var + eps)
    
    # Compute scale and bias for batch norm
    scale = weight / std
    bias_adj = bias - mean * weight / std
    
    # Compute bilinear interpolation scale factors (8x8 -> 64x64)
    scale_h = (orig_H - 1.0) / (H - 1.0) if H > 1 else 0.0
    scale_w = (orig_W - 1.0) / (W - 1.0) if W > 1 else 0.0
    
    # Process each output pixel
    for h_idx in range(H):
        for w_idx in range(W):
            # Calculate source coordinates for bilinear interpolation
            src_h = h_idx * scale_h
            src_w = w_idx * scale_w
            
            h0 = tl.floor(src_h).to(tl.int32)
            w0 = tl.floor(src_w).to(tl.int32)
            h1 = h0 + 1
            w1 = w0 + 1
            
            # Clamp to valid range
            h0 = tl.minimum(tl.maximum(h0, 0), orig_H - 1)
            h1 = tl.minimum(tl.maximum(h1, 0), orig_H - 1)
            w0 = tl.minimum(tl.maximum(w0, 0), orig_W - 1)
            w1 = tl.minimum(tl.maximum(w1, 0), orig_W - 1)
            
            # Interpolation weights
            h_ratio = src_h - tl.floor(src_h)
            w_ratio = src_w - tl.floor(src_w)
            
            # Load 4 corners from input tensor
            idx_h0w0 = n * C * orig_H * orig_W + c * orig_H * orig_W + h0 * orig_W + w0
            idx_h0w1 = n * C * orig_H * orig_W + c * orig_H * orig_W + h0 * orig_W + w1
            idx_h1w0 = n * C * orig_H * orig_W + c * orig_H * orig_W + h1 * orig_W + w0
            idx_h1w1 = n * C * orig_H * orig_W + c * orig_H * orig_W + h1 * orig_W + w1
            
            v_h0w0 = tl.load(input_ptr + idx_h0w0)
            v_h0w1 = tl.load(input_ptr + idx_h0w1)
            v_h1w0 = tl.load(input_ptr + idx_h1w0)
            v_h1w1 = tl.load(input_ptr + idx_h1w1)
            
            # Bilinear interpolation
            interp_val = (1.0 - h_ratio) * (1.0 - w_ratio) * v_h0w0 + \
                        (1.0 - h_ratio) * w_ratio * v_h0w1 + \
                        h_ratio * (1.0 - w_ratio) * v_h1w0 + \
                        h_ratio * w_ratio * v_h1w1
            
            # Load from other tensor (already at target size)
            other_idx = n * C * H * W + c * H * W + h_idx * W + w_idx
            other_val = tl.load(other_ptr + other_idx)
            
            # Add
            sum_val = interp_val + other_val
            
            # Apply batch norm
            bn_val = sum_val * scale + bias_adj
            
            # Apply ReLU
            relu_val = tl.maximum(bn_val, 0.0)
            
            # Store result
            out_idx = n * C * H * W + c * H * W + h_idx * W + w_idx
            tl.store(output_ptr + out_idx, relu_val)


@torch.fx.wrap
def fused_interpolate_add_bn_relu_wrapper(
    tmp_6, in_7, mean, var, weight, bias, eps=1e-05
):
    # Input shapes:
    # tmp_6: (N, C, 8, 8)
    # in_7: (N, C, 64, 64)
    # mean, var, weight, bias: (C,)
    # Output: (N, C, 64, 64)
    
    N, C, orig_H, orig_W = tmp_6.shape
    _, _, H, W = in_7.shape
    
    output = torch.empty((N, C, H, W), device=tmp_6.device, dtype=tmp_6.dtype)
    
    # Grid: (N * C,)
    grid = (N * C,)
    
    fused_interpolate_add_bn_relu_kernel[grid](
        tmp_6, in_7, mean, var, weight, bias,
        output,
        N, C, H, W, orig_H, orig_W,
        eps,
        BLOCK_SIZE=1,
    )
    
    return output


def replacement_func():
    return fused_interpolate_add_bn_relu_wrapper