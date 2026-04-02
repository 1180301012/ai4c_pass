import torch
import triton
import triton.language as tl

# Pattern matching function for fusion of addition + view + softmax + views
def pattern(x, y):
    # Addition with broadcasting: [1, 8, 300, 625] + [1, 1, 300, 625]
    tmp_0 = x + y
    # View operation: [1, 8, 300, 625] -> [8, 300, 625] 
    tmp_1 = tmp_0.view(8, 300, 625)
    # Softmax operation on last dimension
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    # View back to [1, 8, 300, 625] (this is tmp_3, second return value)
    tmp_3 = tmp_2.view(1, 8, 300, 625)
    # View again to [8, 300, 625] (this is tmp_4 before dropout)
    tmp_4 = tmp_3.view(8, 300, 625)
    # Dropout with p=0.0 is identity, so tmp_5 = tmp_4 (first return value)
    tmp_5 = tmp_4
    return tmp_5, tmp_3

# Argument extraction function 
def replacement_args(x, y):
    return (x, y)

# Optimized kernel that fuses addition, view, and softmax
@triton.jit
def fused_add_view_softmax_kernel(
    x_ptr, 
    y_ptr, 
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Load elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors with broadcasting
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # fused operation: addition + exponentiation + normalization for softmax
    out = x + y
    max_val = tl.max(out, axis=-1, keep_dims=True)
    exp_out = tl.exp(out - max_val)
    sum_exp = tl.sum(exp_out, axis=-1, keep_dims=True)
    softmax_out = exp_out / (sum_exp + 1e-20)
    
    tl.store(out_ptr + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def fused_add_view_softmax(x, y):
    # Calculate total elements considering broadcast pattern
    # x shape: [1, 8, 300, 625], y shape: [1, 1, 300, 625]
    # After broadcasting, result is [1, 8, 300, 625]
    total_elements = 1 * 8 * 300 * 625
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # First output: [8, 300, 625] (corresponds to tmp_5)
    out_8_300_625 = torch.empty(8, 300, 625, dtype=x.dtype, device=x.device)
    
    # Second output: [1, 8, 300, 625] (corresponds to tmp_3)
    out_1_8_300_625 = torch.empty(1, 8, 300, 625, dtype=x.dtype, device=x.device)
    
    fused_add_view_softmax_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y, 
        out_ptr=out_1_8_300_625,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # The first output is just a view of the second output
    # tmp_5 corresponds to out_1_8_300_625.view(8, 300, 625)
    # tmp_3 corresponds to out_1_8_300_625
    
    return out_1_8_300_625.view(8, 300, 625), out_1_8_300_625

# Replacement function (no arguments, returns function reference
def replacement_func():
    return fused_add_view_softmax