import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact LayerNorm computation from model.py
# Args are (in_0: bias[768], in_1: weight[768], in_2: tensor[B,S,H], in_3: tensor[B,S,H])
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the full LayerNorm computation pattern from the model.
    """
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15

# Argument extraction function - reorder to (x, y, weight, bias)
def replacement_args(in_0, in_1, in_2, in_3):
    # in_2 and in_3 are 3D tensors, in_0 and in_1 are 1D bias/weight
    return (in_3, in_2, in_1, in_0)

@triton.jit
def fused_layer_norm_kernel(
    x_ptr, y_ptr, bias_ptr, weight_ptr, output_ptr,
    x_stride_b, x_stride_s, x_stride_h,
    y_stride_b, y_stride_s, y_stride_h,
    out_stride_b, out_stride_s, out_stride_h,
    batch_size, seq_len, hidden_dim
):
    """
    Fused LayerNorm kernel: computes add + layer_norm + affine transform.
    Uses 2D grid where each program handles one (batch, seq) position.
    """
    # Program ID
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    # Offsets for hidden dimension
    offs = tl.arange(0, 1024)
    mask = offs < hidden_dim
    
    # Load x and y tensors
    x_offs = pid_b * x_stride_b + pid_s * x_stride_s + offs * x_stride_h
    y_offs = pid_b * y_stride_b + pid_s * y_stride_s + offs * y_stride_h
    
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + y_offs, mask=mask, other=0.0).to(tl.float32)
    
    # Add tensors
    data = x + y
    
    # Compute mean
    sum_data = tl.sum(data)
    mean = sum_data / hidden_dim
    
    # Compute centered values
    centered = data - mean
    
    # Compute variance
    sq = centered * centered
    sum_sq = tl.sum(sq)
    var = sum_sq / hidden_dim
    
    # Normalize
    eps = 1e-07
    std = tl.sqrt(var + eps)
    normalized = centered / std
    
    # Load weight and bias
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Apply affine transform
    output = normalized * w + b
    
    # Store output
    out_offs = pid_b * out_stride_b + pid_s * out_stride_s + offs * out_stride_h
    tl.store(output_ptr + out_offs, output, mask=mask)


@torch.fx.wrap
def fused_layer_norm(x, y, bias, weight):
    """
    Wrapper for fused LayerNorm kernel.
    """
    B, S, H = x.shape
    output = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
    
    grid = (B, S)
    fused_layer_norm_kernel[grid](
        x, y, bias, weight, output,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        B, S, H
    )
    
    return output


def replacement_func():
    return fused_layer_norm