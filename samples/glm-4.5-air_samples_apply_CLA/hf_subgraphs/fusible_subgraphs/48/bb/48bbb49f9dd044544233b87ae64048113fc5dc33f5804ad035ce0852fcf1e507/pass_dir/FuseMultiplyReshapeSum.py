import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """Match the computation pattern - simplified version."""
    # Softmax
    tmp_2 = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    
    # Branch 1: mul + reshape + sum
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(-1, 17, 4096)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    
    # Branch 2: mul + reshape + sum  
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(-1, 17, 4096)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    
    # Concatenate
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    
    return tmp_3, tmp_10


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement kernel."""
    return (in_0, in_1, in_2)


@triton.jit
def softmax_and_weighted_sum_kernel(
    in_2_ptr, weight_0_ptr, weight_1_ptr, 
    out_3_ptr, out_10_ptr,
    B: tl.constexpr, C: tl.constexpr
):
    """Fully fused kernel: softmax + multiply + reshape + sum.
    
    Each program handles one (batch, keypoint) pair.
    Computes softmax over the 64x64 block, then computes both weighted sums.
    """
    batch_idx = tl.program_id(0)
    keypoint_idx = tl.program_id(1)
    
    # Load weights
    w0 = tl.load(weight_0_ptr + tl.arange(0, 64))  # (64,)
    w1 = tl.load(weight_1_ptr + tl.arange(0, 64))  # (64,)
    
    # Compute softmax: find max for numerical stability
    base = batch_idx * C * 4096 + keypoint_idx * 4096
    
    # Find max value in the 64x64 block
    max_val = -float('inf')
    for i in range(64):
        offset = base + i * 64 + tl.arange(0, 64)
        vals = tl.load(in_2_ptr + offset)
        max_val = tl.max(tl.maximum(max_val, vals))
    
    # Compute exp sum
    exp_sum = 0.0
    for i in range(64):
        offset = base + i * 64 + tl.arange(0, 64)
        vals = tl.load(in_2_ptr + offset)
        exp_vals = tl.exp(vals - max_val)
        exp_sum += tl.sum(exp_vals)
    
    # Compute weighted sums while computing softmax
    sum_0 = 0.0
    sum_1 = 0.0
    
    for i in range(64):
        offset = base + i * 64 + tl.arange(0, 64)
        vals = tl.load(in_2_ptr + offset)
        exp_vals = tl.exp(vals - max_val)
        softmax_vals = exp_vals / exp_sum
        
        # Store softmax result for tmp_3 output
        out_3_offset = batch_idx * C * 4096 + keypoint_idx * 4096 + i * 64
        tl.store(out_3_ptr + out_3_offset + tl.arange(0, 64), softmax_vals)
        
        # Compute weighted sums
        # sum_0: sum over j of softmax[i,j] * w0[j]
        sum_0 += tl.sum(softmax_vals * w0)
        
        # sum_1: sum over i of softmax[i,j] * w1[i]
        sum_1 += tl.sum(softmax_vals * w1[i])
    
    # Store weighted sum results for tmp_10 output
    out_10_offset = batch_idx * C * 2 + keypoint_idx * 2
    tl.store(out_10_ptr + out_10_offset, sum_0)
    tl.store(out_10_ptr + out_10_offset + 1, sum_1)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """Pure Triton implementation - fully fuses softmax + mul + reshape + sum.
    
    Returns:
    - tmp_3: softmax reshaped to (-1, 17, 64, 64)
    - tmp_10: concatenated weighted sums (-1, 17, 2)
    """
    B = in_2.shape[0]
    C = in_2.shape[1]  # 17
    
    # Allocate outputs
    out_3 = torch.empty((B * C * 4096,), dtype=torch.float32, device=in_2.device)
    out_10 = torch.empty((B, C * 2), dtype=torch.float32, device=in_2.device)
    
    # Flatten weights for Triton kernel
    w0_flat = in_0.reshape(-1)  # (64,)
    w1_flat = in_1.reshape(-1)  # (64,)
    
    # Launch fused kernel
    grid = (B, C)
    softmax_and_weighted_sum_kernel[grid](
        in_2, w0_flat, w1_flat,
        out_3, out_10,
        B, C
    )
    
    # Reshape outputs to match original
    tmp_3 = out_3.reshape(B, C, 64, 64)
    tmp_10 = out_10.reshape(B, C, 2)
    
    return tmp_3, tmp_10


def replacement_func():
    return fused_kernel_wrapper