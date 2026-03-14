import torch
import triton
import triton.language as tl

# Pattern: tmp_0 = 0 + in_1; tmp_0 += in_0; tmp_1 = tmp_0; tmp_2 = tmp_1.mean((2,3), keepdim=True)
# Returns (tmp_1, tmp_2)

def pattern(in_0, in_1):
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['HW'],
)
@triton.jit
def fused_add2_mean_kernel(
    in_0_ptr, in_1_ptr,
    out_sum_ptr, out_mean_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    bc_idx = tl.program_id(0)
    base_offset = bc_idx * HW
    
    acc = tl.zeros([1], dtype=tl.float32)
    
    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        global_offsets = base_offset + offsets
        
        x0 = tl.load(in_0_ptr + global_offsets, mask=mask, other=0.0)
        x1 = tl.load(in_1_ptr + global_offsets, mask=mask, other=0.0)
        
        # 0 + in_1 + in_0 = in_0 + in_1
        s = x0 + x1
        
        tl.store(out_sum_ptr + global_offsets, s, mask=mask)
        acc += tl.sum(s, axis=0)
    
    mean = acc / HW
    tl.store(out_mean_ptr + bc_idx, mean)

@torch.fx.wrap
def fused_add2_mean(in_0, in_1):
    B, C, H, W = in_0.shape
    HW = H * W
    BC = B * C
    
    out_sum = torch.empty_like(in_0)
    out_mean = torch.empty(BC, device=in_0.device, dtype=in_0.dtype)
    
    grid = (BC,)
    
    fused_add2_mean_kernel[grid](
        in_0, in_1,
        out_sum, out_mean,
        HW,
    )
    
    out_mean = out_mean.view(B, C, 1, 1)
    return (out_sum, out_mean)

def replacement_func():
    return fused_add2_mean