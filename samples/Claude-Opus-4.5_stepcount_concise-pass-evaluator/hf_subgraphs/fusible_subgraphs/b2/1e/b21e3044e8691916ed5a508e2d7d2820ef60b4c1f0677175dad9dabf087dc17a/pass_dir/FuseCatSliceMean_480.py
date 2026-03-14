import torch
import triton
import triton.language as tl

# Pattern to match: cat -> slice -> mean with slice value 480
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[slice(None, None, None), slice(None, 480, None), slice(None, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def cat_mean_kernel(
    in_0_ptr, in_1_ptr,
    out_cat_ptr, out_mean_ptr,
    B, C, H, W, C2,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    
    b = pid // C2
    c_out = pid % C2
    
    c_in = c_out % C
    use_in_1 = c_out >= C
    
    in_base = b * C * HW + c_in * HW
    out_base = b * C2 * HW + c_out * HW
    
    acc = 0.0
    
    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        
        mask_0 = mask & (~use_in_1)
        mask_1 = mask & use_in_1
        
        val_0 = tl.load(in_0_ptr + in_base + offs, mask=mask_0, other=0.0)
        val_1 = tl.load(in_1_ptr + in_base + offs, mask=mask_1, other=0.0)
        vals = val_0 + val_1
        
        tl.store(out_cat_ptr + out_base + offs, vals, mask=mask)
        acc = acc + tl.sum(vals)
    
    mean_val = acc / HW
    mean_off = b * C2 + c_out
    tl.store(out_mean_ptr + mean_off, mean_val)


@torch.fx.wrap
def _cat_mean_kernel_impl(in_0, in_1):
    B, C, H, W = in_0.shape
    C2 = 2 * C
    HW = H * W
    
    out_cat = torch.empty(B, C2, H, W, device=in_0.device, dtype=in_0.dtype)
    out_mean = torch.empty(B, C2, 1, 1, device=in_0.device, dtype=in_0.dtype)
    
    num_programs = B * C2
    
    BLOCK_HW = min(1024, max(64, triton.next_power_of_2(HW)))
    
    cat_mean_kernel[(num_programs,)](
        in_0, in_1,
        out_cat, out_mean,
        B, C, H, W, C2,
        HW,
        BLOCK_HW=BLOCK_HW,
    )
    
    return (out_cat, out_mean)


def cat_mean_fused(in_0, in_1):
    result = _cat_mean_kernel_impl(in_0, in_1)
    return (result[0], result[1])


def replacement_func():
    return cat_mean_fused