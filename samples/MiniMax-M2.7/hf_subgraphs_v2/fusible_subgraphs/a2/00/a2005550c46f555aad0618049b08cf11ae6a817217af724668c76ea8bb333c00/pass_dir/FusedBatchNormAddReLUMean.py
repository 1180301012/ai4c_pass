import torch
import triton
import triton.language as tl

@triton.jit
def fused_bn_add_relu_mean_kernel(
    in_4_ptr, in_5_ptr, in_0_ptr, in_1_ptr, in_3_ptr, in_2_ptr,
    out_ptr, mean_ptr,
    B, C, H, W,
    stride_in4, stride_in5,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for: batch_norm -> add -> relu -> mean
    
    Each program processes one (batch, channel) position and iterates over H*W spatial dims.
    
    batch_norm: out = (x - mean) / sqrt(var + eps) * weight + bias
    add: out = out + residual
    relu: out = max(0, out)
    mean: spatial average with keepdim=True
    """
    pid = tl.program_id(0)
    
    # Total spatial elements per (B, C) position
    n_elements_per_bc = H * W
    
    # Compute B, C, h, w indices for this program
    b = pid // C
    c = pid % C
    
    # Offsets for this program (each handles H*W spatial positions)
    offs_h = tl.arange(0, H)
    offs_w = tl.arange(0, W)
    
    # Accumulator for spatial mean
    sum_val = 0.0
    
    # Iterate over spatial dimensions
    for h_idx in range(H):
        for w_idx in range(W):
            # Flattened offset for feature tensor
            off = (((b * C + c) * H + h_idx) * W + w_idx)
            
            # Load input feature (in_4)
            x = tl.load(in_4_ptr + off * stride_in4).to(tl.float32)
            
            # Load batch norm parameters
            running_mean = tl.load(in_0_ptr + c).to(tl.float32)
            running_var = tl.load(in_1_ptr + c).to(tl.float32)
            weight = tl.load(in_3_ptr + c).to(tl.float32)
            bias = tl.load(in_2_ptr + c).to(tl.float32)
            
            # Batch norm: normalize
            inv_std = tl.rsqrt(running_var + 1e-05)
            bn_out = (x - running_mean) * inv_std * weight + bias
            
            # Load residual (in_5) and add
            residual = tl.load(in_5_ptr + off * stride_in5).to(tl.float32)
            out = bn_out + residual
            
            # ReLU
            out = tl.where(out > 0, out, 0.0)
            
            # Accumulate for mean
            sum_val += out
            
            # Store output
            tl.store(out_ptr + off, out)
    
    # Compute mean and store
    mean_val = sum_val / n_elements_per_bc
    mean_off = b * C + c
    tl.atomic_add(mean_ptr + mean_off, mean_val)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern: batch_norm -> add -> relu -> mean
    
    This matches the GhostNet-style residual block computation.
    
    Args:
        in_0: running_mean [C]
        in_1: running_var [C]
        in_2: bias [C]
        in_3: weight [C]
        in_4: input tensor [B, C, H, W]
        in_5: residual tensor [B, C, H, W]
    
    Returns:
        tmp_6: relu output [B, C, H, W]
        tmp_7: spatial mean [B, C, 1, 1]
    """
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@torch.fx.wrap
def fused_bn_add_relu_mean(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Fused kernel wrapper for batch_norm + add + relu + mean.
    
    Reduces kernel launches from 4 to 1 and eliminates intermediate tensor
    allocations in global memory.
    """
    B, C, H, W = in_4.shape
    
    # Output tensors
    out = torch.empty((B, C, H, W), device=in_4.device, dtype=in_4.dtype)
    mean_out = torch.empty((B, C, 1, 1), device=in_4.device, dtype=in_4.dtype)
    
    # Grid: one program per (B, C) position
    n_programs = B * C
    
    # Launch fused kernel
    fused_bn_add_relu_mean_kernel[(n_programs,)](
        in_4, in_5, in_0, in_1, in_3, in_2,
        out, mean_out,
        B, C, H, W,
        1, 1,  # strides (contiguous)
        B * C * H * W,
        128  # BLOCK_SIZE
    )
    
    return (out, mean_out)


def replacement_func():
    return fused_bn_add_relu_mean