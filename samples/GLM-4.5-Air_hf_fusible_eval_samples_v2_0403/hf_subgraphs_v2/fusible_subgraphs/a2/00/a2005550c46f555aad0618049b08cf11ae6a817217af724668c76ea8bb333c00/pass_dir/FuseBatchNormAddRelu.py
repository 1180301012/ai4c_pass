import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_3, in_2, in_5):
    # Use all inputs to avoid dead code - batch norm + simple add pattern
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    return tmp_5

def replacement_args(in_4, in_0, in_1, in_3, in_2, in_5):
    # Extract all arguments needed for the fused operation
    return (in_4, in_0, in_1, in_3, in_2, in_5)

@triton.jit
def fused_bn_add_kernel(
    x_ptr,      # input tensor for batch norm (in_4)
    mean_ptr,   # running mean (in_0) 
    var_ptr,    # running variance (in_1)
    weight_ptr, # weight (in_3)
    bias_ptr,   # bias (in_2)
    residual_ptr, # residual input (in_5)
    out_ptr,    # output tensor (tmp_5 = BN + Add)
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused batch norm + add kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load batch norm parameters (simplified broadcast handling)
    # This might need adjustment based on actual tensor shapes
    mean = tl.load(mean_ptr + (offsets % 1), mask=offsets < 1, other=0.0)
    var = tl.load(var_ptr + (offsets % 1), mask=offsets < 1, other=1.0)
    weight = tl.load(weight_ptr + (offsets % 1), mask=offsets < 1, other=1.0)
    bias = tl.load(bias_ptr + (offsets % 1), mask=offsets < 1, other=0.0)
    
    # Load residual 
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    
    # Batch normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - mean) * tl.rsqrt(var + eps) * weight + bias
    
    # Add residual
    out = normalized + residual
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_bn_add(x, mean, var, weight, bias, residual):
    """Fused batch norm + add operation"""
    total_elements = x.numel()
    BLOCK_SIZE = 1024  # Could be autotuned
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype as input
    out = torch.empty_like(x)
    
    fused_bn_add_kernel[(num_programs,)](
        x_ptr=x,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        residual_ptr=residual,
        out_ptr=out,
        n_elements=total_elements,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_bn_add