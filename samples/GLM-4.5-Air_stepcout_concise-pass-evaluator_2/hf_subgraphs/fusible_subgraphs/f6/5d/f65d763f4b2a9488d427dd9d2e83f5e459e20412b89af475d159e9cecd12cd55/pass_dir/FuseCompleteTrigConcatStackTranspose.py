import torch
import triton
import triton.language as tl

@triton.jit
def fused_trigonometric_kernel(
    x_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that computes both cosine and sine simultaneously"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both trig functions in parallel
    cos_vals = tl.cos(x)
    sin_vals = tl.sin(x)
    
    # Store results
    tl.store(cos_out_ptr + offsets, cos_vals, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_vals, mask=mask)

@torch.fx.wrap
def fused_trigonometric(x):
    """Fused function that computes both cosine and sine of the same tensor"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    cos_out = torch.empty_like(x)
    sin_out = torch.empty_like(x)
    
    fused_trigonometric_kernel[(num_programs,)](
        x, cos_out, sin_out, n_elements, BLOCK_SIZE
    )
    
    return cos_out, sin_out

def pattern(in0, in1, in2):
    """Pattern that matches the trigonometric operations specifically"""
    tmp_0 = torch.cat((in0, in2), dim=-1)
    # Match the separate cos and sin operations
    tmp_1 = in1.cos()
    tmp_2 = in1.sin()
    tmp_3 = torch.cat((tmp_1, tmp_2), dim=-1)
    tmp_1 = tmp_2 = None
    tmp_4 = torch.stack((tmp_0, tmp_3), dim=-1)
    tmp_0 = tmp_3 = None
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_4 = None
    return (tmp_5,)

def replacement_args(in0, in1, in2):
    return (in0, in1, in2)

def replacement_func():
    """Replacement function that uses fused trigonometric computation"""
    def optimized_forward(in0, in1, in2):
        # Replace separate cos/sin with fused operation
        tmp_1, tmp_2 = fused_trigonometric(in1)
        
        # Keep the rest of the computation the same
        tmp_0 = torch.cat((in0, in2), dim=-1)
        tmp_3 = torch.cat((tmp_1, tmp_2), dim=-1)
        tmp_4 = torch.stack((tmp_0, tmp_3), dim=-1)
        tmp_5 = tmp_4.transpose(-1, -2)
        return (tmp_5,)
    
    return optimized_forward