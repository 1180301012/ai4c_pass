import torch
import triton
import triton.language as tl


@triton.jit
def weighted_sum_norm_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements_out,
    reduction_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. Cast in_0 to float32
    2. Multiply in_1 * cast(in_0)
    3. Dual reduction: sum along dim=1 for both tmp_1 and tmp_0
    4. Clamp denominator with min=1e-09
    5. Divide numerator by clamped denominator
    6. Unsqueeze to [1, 1, 1024]
    
    Input shapes: in_0 [1, 10, 1024], in_1 [1, 10, 1024]
    Output shape: [1, 1, 1024]
    """
    pid = tl.program_id(0)
    
    # Offset in output (dim 2 of original tensor)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_out
    
    # Initialize accumulators for dual reduction
    acc_in_1_times_in_0 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc_in_0 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Reduction loop over dim=1 (size 10)
    for i in range(reduction_size):
        # Base offset for this reduction iteration
        base_offset = i * n_elements_out
        
        # Load in_0 (int64) - load as int64 then convert
        in_0_val = tl.load(in_0_ptr + base_offset + offsets, mask=mask, other=0.0)
        # Cast to float32
        in_0_float = in_0_val.to(tl.float32)
        
        # Load in_1 (bfloat16/float16)
        in_1_val = tl.load(in_1_ptr + base_offset + offsets, mask=mask, other=0.0)
        # Cast to float32 for computation
        in_1_float = in_1_val.to(tl.float32)
        
        # Accumulate
        acc_in_1_times_in_0 += in_1_float * in_0_float
        acc_in_0 += in_0_float
    
    # Clamp denominator with min=1e-09 (using broadcast of scalar constant)
    min_val = 1e-09
    acc_in_0 = tl.where(acc_in_0 < min_val, min_val, acc_in_0)
    
    # Compute result
    result = acc_in_1_times_in_0 / acc_in_0
    
    # Unsqueeze along dim 1 - output is [1, 1, n_elements_out]
    # Store to output with output_ptr pointing to [0, 0, 0]
    tl.store(out_ptr + offsets, result, mask=mask)


def pattern(in_0, in_1):
    """
    Match the computation pattern:
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6
    """
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def fused_weighted_sum_norm(in_0, in_1):
    """
    Wrapper for the fused kernel that computes:
    result = sum(in_1 * cast(in_0)) / max(sum(cast(in_0)), 1e-09)
    """
    # Input shape: [1, 10, 1024]
    B, reduction_dim, N = in_0.shape
    
    # Output shape after sum and unsqueeze: [1, 1, 1024]
    output_elements = N
    
    # Allocate output tensor
    out = torch.empty((1, 1, N), dtype=torch.float32, device=in_0.device)
    
    # Block size for reduction - using 512 for best performance
    BLOCK_SIZE = 512
    
    # Grid size = number of output elements
    grid = (output_elements,)
    
    # Launch kernel
    weighted_sum_norm_kernel[grid](
        in_0,
        in_1,
        out,
        output_elements,
        reduction_dim,
        BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_weighted_sum_norm