import torch
import triton
import triton.language as tl

# Pattern matching function - must match the model exactly
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.2, False, False)
    return tmp_3

# Extract arguments for replacement
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Multi-block kernel for better parallelism
@triton.jit
def cat_flatten_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    total = C0 + C1 + C2 + C3
    
    b1 = C0
    b2 = C0 + C1
    b3 = C0 + C1 + C2
    
    m0 = offs < b1
    m1 = (offs >= b1) & (offs < b2)
    m2 = (offs >= b2) & (offs < b3)
    m3 = (offs >= b3) & (offs < total)
    
    v0 = tl.load(in_0_ptr + offs, mask=m0, other=0.0)
    v1 = tl.load(in_1_ptr + (offs - b1), mask=m1, other=0.0)
    v2 = tl.load(in_2_ptr + (offs - b2), mask=m2, other=0.0)
    v3 = tl.load(in_3_ptr + (offs - b3), mask=m3, other=0.0)
    
    result = v0 + v1 + v2 + v3
    tl.store(out_ptr + offs, result, mask=offs < total)

@torch.fx.wrap
def cat_flatten_fused(in_0, in_1, in_2, in_3):
    C0 = in_0.shape[1]
    C1 = in_1.shape[1]
    C2 = in_2.shape[1]
    C3 = in_3.shape[1]
    total_C = C0 + C1 + C2 + C3
    
    out = torch.empty((1, total_C), dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE = 256
    grid = (total_C + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    cat_flatten_kernel[(grid,)](
        in_0.view(-1), in_1.view(-1), in_2.view(-1), in_3.view(-1),
        out.view(-1),
        C0, C1, C2, C3,
        BLOCK_SIZE,
        num_warps=4,
    )
    
    return out

def replacement_func():
    return cat_flatten_fused