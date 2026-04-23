import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return (tmp_4,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_mask_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    x_f = x.to(tl.float32)
    val = 1.0 - x_f
    fill_val = -3.4028234663852886e+38
    result = tl.where(val != 0.0, fill_val * val, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_mask_op(in_0):
    n_elements = in_0.numel()
    BLOCK_SIZE = 512
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty(in_0.shape, dtype=torch.float32, device=in_0.device)

    fused_mask_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )

    return out

def replacement_func():
    return fused_mask_op