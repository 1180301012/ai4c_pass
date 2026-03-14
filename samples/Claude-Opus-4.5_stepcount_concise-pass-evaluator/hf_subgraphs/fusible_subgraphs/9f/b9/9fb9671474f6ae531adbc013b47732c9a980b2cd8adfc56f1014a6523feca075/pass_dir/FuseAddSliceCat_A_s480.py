import torch
import triton
import triton.language as tl

# Pattern A: in_0 + in_1, slice in_2 from channel 480
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 + in_1
    tmp_1 = in_2[slice(None, None, None), slice(480, None, None)]
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def add_slice_cat_kernel_480(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    N, C_add, C_slice, H, W,
    slice_start,
    stride_0_n, stride_0_c, stride_0_h, stride_0_w,
    stride_2_n, stride_2_c, stride_2_h, stride_2_w,
    BLOCK_SIZE: tl.constexpr,
):
    C_out = C_add + C_slice
    total_elements = N * C_out * H * W
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    w_idx = offsets % W
    temp = offsets // W
    h_idx = temp % H
    temp = temp // H
    c_idx = temp % C_out
    n_idx = temp // C_out
    
    is_add = c_idx < C_add
    
    add_offset = n_idx * stride_0_n + c_idx * stride_0_c + h_idx * stride_0_h + w_idx * stride_0_w
    
    slice_c = c_idx - C_add + slice_start
    slice_offset = n_idx * stride_2_n + slice_c * stride_2_c + h_idx * stride_2_h + w_idx * stride_2_w
    
    in_0_val = tl.load(in_0_ptr + add_offset, mask=mask & is_add, other=0.0)
    in_1_val = tl.load(in_1_ptr + add_offset, mask=mask & is_add, other=0.0)
    add_result = in_0_val + in_1_val
    
    slice_val = tl.load(in_2_ptr + slice_offset, mask=mask & (~is_add), other=0.0)
    
    result = tl.where(is_add, add_result, slice_val)
    
    out_offset = n_idx * (C_out * H * W) + c_idx * (H * W) + h_idx * W + w_idx
    tl.store(out_ptr + out_offset, result, mask=mask)

@torch.fx.wrap
def add_slice_cat_480(in_0, in_1, in_2):
    N, C_add, H, W = in_0.shape
    slice_start = 480
    C_slice = in_2.shape[1] - slice_start
    C_out = C_add + C_slice
    
    out = torch.empty((N, C_out, H, W), dtype=in_0.dtype, device=in_0.device)
    
    total_elements = N * C_out * H * W
    BLOCK_SIZE = 1024
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    s0 = in_0.stride()
    s2 = in_2.stride()
    
    add_slice_cat_kernel_480[(num_blocks,)](
        in_0, in_1, in_2, out,
        N, C_add, C_slice, H, W,
        slice_start,
        s0[0], s0[1], s0[2], s0[3],
        s2[0], s2[1], s2[2], s2[3],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return add_slice_cat_480