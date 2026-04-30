import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_rope_compute_kernel(
    in_2_ptr, in_1_ptr, in_4_ptr,
    out_ptr,
    stride_3,
    n_elements, HALF_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all inputs
    in_2_val = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_4_val = tl.load(in_4_ptr + offsets, mask=mask, other=0.0)
    
    # Compute base: in_2 * in_1
    base = in_2_val * in_1_val
    
    # Compute rotated value using conditional load
    # For d < 128: use in_2[d+128] negated, for d >= 128: use in_2[d-128]
    d = offsets % stride_3
    neg_offset = offsets + HALF_SIZE
    pos_offset = offsets - HALF_SIZE
    
    in_2_neg = tl.load(in_2_ptr + neg_offset, mask=mask, other=0.0)
    in_2_pos = tl.load(in_2_ptr + pos_offset, mask=mask, other=0.0)
    
    is_first_half = d < HALF_SIZE
    rotated = tl.where(is_first_half, -in_2_neg, in_2_pos)
    
    # Final computation
    out_val = base + rotated * in_4_val
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_rope_compute(in_2, in_1, in_4):
    n_elements = in_2.numel()
    num_programs = (n_elements + 127) // 128
    
    out = torch.empty_like(in_2)
    
    fused_rope_compute_kernel[(num_programs,)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_4_ptr=in_4,
        out_ptr=out,
        stride_3=in_2.stride(3),
        n_elements=n_elements,
        HALF_SIZE=128,
    )
    
    return out

def pattern(in_2, in_1, in_4):
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_2 = None
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_3 = tmp_1 = None
    tmp_5 = tmp_4 * in_4
    tmp_4 = None
    tmp_6 = tmp_0 + tmp_5
    tmp_0 = tmp_5 = None
    return tmp_6

def replacement_args(in_2, in_1, in_4):
    return (in_2, in_1, in_4)

def replacement_func():
    return fused_rope_compute