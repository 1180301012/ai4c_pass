import torch
import triton
import triton.language as tl



@triton.jit
def fused_exp_mul_add_kernel(
    in_0_ptr,           # scalar bias of shape [1]
    in_1_ptr,           # scalar scale of shape [1] 
    in_2_ptr,           # tensor [2,1]
    out_ptr,            # output tensor [2,1]
    N_elements: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid == 0:  # Single program for small tensor
        # Load all scalars at once
        in_0_val = tl.load(in_0_ptr + 0)
        in_1_val = tl.load(in_1_ptr + 0)
        
        # Fused computation: y + x * exp(a)
        exp_scale = tl.exp(in_1_val)
        
        # Load and compute result
        offsets = tl.arange(0, N_elements)
        mask = offsets < N_elements
        x = tl.load(in_2_ptr + offsets, mask=mask)
        result = in_0_val + x * exp_scale
        
        # Store result
        tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_exp_mul_add(in_0, in_1, in_2):
    out = torch.empty_like(in_2)
    
    # Single kernel handles the computation
    fused_exp_mul_add_kernel[(1,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        N_elements=2,
    )
    
    return out

def pattern(in_0, in_1, in_2):
    # Match only the fused computation: exp + multiply + add
    tmp_0 = in_1.exp()
    tmp_1 = in_2 * tmp_0
    tmp_2 = tmp_1 + in_0
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return fused_exp_mul_add