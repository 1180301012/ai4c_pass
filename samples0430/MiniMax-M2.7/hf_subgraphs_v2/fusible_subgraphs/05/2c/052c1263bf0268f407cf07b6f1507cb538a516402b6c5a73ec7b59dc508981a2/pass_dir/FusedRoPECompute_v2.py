import torch
import triton
import triton.language as tl

@triton.jit
def fused_rope_compute_kernel(
    in_2_ptr, in_1_ptr, in_4_ptr,
    out_ptr,
    stride_1, stride_2, stride_3,
    n_elements, HALF_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # For shape [1, 1, 3, 256] with strides [768, 768, 256, 1]:
    # flat_index = b*768 + h*768 + s*256 + d = s*256 + d
    s = offsets // stride_3  # = s * 256 // 256 = s
    d = offsets % stride_3   # = d
    
    # Load in_2[d] and in_1[d]
    in_2_val = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    tmp_0 = in_2_val * in_1_val
    
    # Compute rotated value: for d < 128: -in_2[d+128], for d >= 128: in_2[d-128]
    neg_offset = offsets + HALF_SIZE  # for d in [0,127], this accesses [128,255]
    pos_offset = offsets - HALF_SIZE  # for d in [128,255], this accesses [0,127]
    
    in_2_neg_val = tl.load(in_2_ptr + neg_offset, mask=mask, other=0.0)
    in_2_pos_val = tl.load(in_2_ptr + pos_offset, mask=mask, other=0.0)
    
    # Select based on d < HALF_SIZE
    is_first_half = d < HALF_SIZE
    in_2_rotated = tl.where(is_first_half, -in_2_neg_val, in_2_pos_val)
    
    # Load in_4[d] and compute
    in_4_val = tl.load(in_4_ptr + offsets, mask=mask, other=0.0)
    tmp_5 = in_2_rotated * in_4_val
    out_val = tmp_0 + tmp_5
    
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_rope_compute(in_2, in_1, in_4):
    n_elements = in_2.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_2)
    
    fused_rope_compute_kernel[(num_programs,)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_4_ptr=in_4,
        out_ptr=out,
        stride_1=in_2.stride(1),
        stride_2=in_2.stride(2),
        stride_3=in_2.stride(3),
        n_elements=n_elements,
        HALF_SIZE=128,
        BLOCK_SIZE=BLOCK_SIZE,
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