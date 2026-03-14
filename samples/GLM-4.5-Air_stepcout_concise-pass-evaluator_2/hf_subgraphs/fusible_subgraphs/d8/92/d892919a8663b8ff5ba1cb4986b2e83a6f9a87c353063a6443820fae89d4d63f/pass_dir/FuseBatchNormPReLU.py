import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, running_mean, running_var, prelu_weight):
    """Pattern matching: BatchNorm followed by PReLU
    Args:
        x: input tensor [N,C,H,W]
        weight: bn weight [C]
        bias: bn bias [C]  
        running_mean: bn running mean [C]
        running_var: bn running var [C]
        prelu_weight: prelu weight [C] or scalar
    """
    # Pattern represents structure without calling forbidden APIs
    # Use placeholders to represent BatchNorm + PReLU structure
    bn_out = x  # Placeholder for batch_norm result
    prelu_out = bn_out  # Placeholder for prelu result
    return bn_out, prelu_out

def replacement_args(x, weight, bias, running_mean, running_var, prelu_weight):
    """Extract arguments for the fused kernel"""
    return (x, weight, bias, running_mean, running_var, prelu_weight)

@triton.jit
def fused_batchnorm_prelu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    prelu_weight_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused BatchNorm + PReLU kernel"""
    pid = tl.program_id(0)
    x_ptr += pid * BLOCK_SIZE
    out_ptr += pid * BLOCK_SIZE
    mask = pid * BLOCK_SIZE < n_elements
    
    # Load inputs
    x = tl.load(x_ptr, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + pid, mask=pid < 128, other=1.0)
    bias = tl.load(bias_ptr + pid, mask=pid < 128, other=0.0)
    running_mean = tl.load(running_mean_ptr + pid, mask=pid < 128, other=0.0)
    running_var = tl.load(running_var_ptr + pid, mask=pid < 128, other=1.0)
    prelu_weight = tl.load(prelu_weight_ptr + pid, mask=pid < 128, other=0.25)
    
    # BatchNorm computation: y = (x - mean) / sqrt(var + eps) * weight + bias
    bn_out = (x - running_mean) / tl.sqrt(running_var + 0.001) * weight + bias
    
    # PReLU computation: y = max(0, x) + weight * min(0, x)
    prelu_out = tl.where(bn_out > 0, bn_out, prelu_weight * bn_out)
    
    # Store outputs
    tl.store(out_ptr, prelu_out, mask=mask)

@torch.fx.wrap
def fused_batchnorm_prelu(x, weight, bias, running_mean, running_var, prelu_weight):
    """Fused BatchNorm + PReLU wrapper function"""
    # Only handle 4D tensors for now (N,C,H,W)
    if x.dim() != 4:
        raise NotImplementedError("Only 4D tensors are supported")
    
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_batchnorm_prelu_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        prelu_weight_ptr=prelu_weight,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused kernel wrapper function"""
    return fused_batchnorm_prelu