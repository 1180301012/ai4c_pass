import torch
import triton
import triton.language as tl

def pattern(in_4, in_3, in_0):
    tmp_3 = in_4 / in_3
    tmp_4 = tmp_3.to(torch.float32)
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_4 * tmp_5
    return tmp_6

def replacement_args(in_4, in_3, in_0):
    return (in_4, in_3, in_0)

@triton.jit
def fused_div_type_unsqueeze_mul_kernel(
    out_ptr,
    ptr_4,
    scalar_3_val,
    ptr_0,
    n_elements,
    feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load in_4 (mul_2 output)
    in_4_val = tl.load(ptr_4 + offsets, mask=mask, other=0.0)
    
    # Load in_0 (attention mask) 
    in_0_val = tl.load(ptr_0 + offsets, mask=mask, other=0.0)
    
    # Compute: in_4 / scalar_3_val (type conversion implicit) * in_0.unsqueeze(-1)
    result = (in_4_val / scalar_3_val) * in_0_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_div_type_unsqueeze_mul(in_4, in_3, in_0):
    # Calculate total elements and feature dimension
    if len(in_4.shape) == 3:
        batch_size, seq_len, hidden_dim = in_4.shape
        total_elements = batch_size * seq_len * hidden_dim
    else:
        total_elements = in_4.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_4)
    
    # Extract scalar value from in_3 tensor
    scalar_3_val = in_3.detach().cpu().flatten()[0].item()
    
    fused_div_type_unsqueeze_mul_kernel[(num_programs,)](
        out_ptr=out,
        ptr_4=in_4,
        scalar_3_val=scalar_3_val,
        ptr_0=in_0,
        n_elements=total_elements,
        feature_dim=320,  # From weight_meta.py
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_div_type_unsqueeze_mul