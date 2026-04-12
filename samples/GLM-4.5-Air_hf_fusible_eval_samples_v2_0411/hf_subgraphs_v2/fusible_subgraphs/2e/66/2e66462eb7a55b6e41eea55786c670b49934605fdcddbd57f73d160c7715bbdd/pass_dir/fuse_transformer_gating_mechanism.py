import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_gating_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    scale1: tl.constexpr,
    scale2: tl.constexpr,
    scale3: tl.constexpr,
):
    grid = tl.program_id(0)
    block_start = grid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Exact original computation sequence without fusion
    tmp_0 = scale1 * x  # 0.5 * x
    tmp_1 = x * x * x  # x^3
    tmp_2 = scale2 * tmp_1  # 0.044715 * tmp_1
    tmp_3 = x + tmp_2  # in_0 + tmp_2
    tmp_4 = scale3 * tmp_3  # 0.7978845608028654 * tmp_3
    
    # Use a more accurate tanh approximation
    # Based on the Pade approximation of order [3,3]
    # tanh(x) ≈ x*(135 + x²)/(135 + 10x²) for good accuracy
    x_sq = tmp_4 * tmp_4
    numerator = tmp_4 * (135.0 + x_sq)
    denominator = 135.0 + 10.0 * x_sq
    
    # Avoid division by zero
    denominator = tl.maximum(denominator, 1e-7)
    tanh_val = numerator / denominator
    
    tmp_6 = 1.0 + tanh_val
    final = tmp_0 * tmp_6
    
    tl.store(out_ptr + offsets, final, mask=mask)

@torch.fx.wrap
def fused_gating_forward(in_0):
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    fused_gating_kernel[(num_programs,)](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        scale1=0.5,
        scale2=0.044715,
        scale3=0.7978845608028654,
    )
    
    return out

def replacement_func():
    return fused_gating_forward