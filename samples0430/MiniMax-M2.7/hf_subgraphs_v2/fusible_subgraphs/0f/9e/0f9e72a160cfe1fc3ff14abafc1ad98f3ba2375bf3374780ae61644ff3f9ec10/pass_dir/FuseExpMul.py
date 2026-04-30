import torch
import triton
import triton.language as tl

@triton.jit
def exp_mul_kernel(
    scalar_ptr, tensor_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused exp + multiply kernel for scalar * tensor element-wise."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    tensor = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
    scalar = tl.load(scalar_ptr).to(tl.float32)  # Load scalar and cast to fp32 for exp
    
    # Fused exp and multiply: exp(scalar) * tensor
    exp_scalar = tl.exp(scalar)
    result = tensor * exp_scalar
    
    tl.store(out_ptr + offsets, result, mask=mask)

def pattern(in_0, tmp_4):
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return tmp_6

def replacement_args(in_0, tmp_4):
    return (in_0, tmp_4)

@torch.fx.wrap
def exp_mul_wrapper(in_0, tmp_4):
    N = tmp_4.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(tmp_4)
    
    exp_mul_kernel[(num_programs,)](
        scalar_ptr=in_0,
        tensor_ptr=tmp_4,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return exp_mul_wrapper