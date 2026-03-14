import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """Pattern matching: exp -> mul -> add"""
    tmp_0 = in_1.exp()
    tmp_1 = in_2 * tmp_0
    tmp_2 = tmp_1 + in_0
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    """Extract arguments for replacement function"""
    return (in_0, in_1, in_2)


@triton.jit
def fused_exp_mul_add_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_ptr,
    n_elements: tl.constexpr,
):
    """Fused kernel: exp(logit_scale) * logits_per_text + logit_bias"""
    # Single program handles all elements (tensor is very small)
    offsets = tl.arange(0, n_elements)
    mask = offsets < n_elements
    
    # Load scalars
    logit_bias = tl.load(in_0_ptr)
    logit_scale = tl.load(in_1_ptr)
    exp_scale = tl.exp(logit_scale)
    
    # Load and compute
    logits_val = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    result = tl.fma(logits_val, exp_scale, logit_bias)
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_exp_mul_add_kernel_wrapper(in_0, in_1, in_2):
    """
    Fused kernel: computes:
    tmp_0 = exp(in_1)
    tmp_1 = in_2 * tmp_0
    tmp_2 = tmp_1 + in_0
    Returns tmp_2
    """
    # Get total elements
    n_elements = in_2.numel()
    
    # Allocate output tensor
    out = torch.empty_like(in_2)
    
    # Launch single-program kernel
    fused_exp_mul_add_kernel[(1,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    return out


def replacement_func():
    """Return the replacement function"""
    return fused_exp_mul_add_kernel_wrapper