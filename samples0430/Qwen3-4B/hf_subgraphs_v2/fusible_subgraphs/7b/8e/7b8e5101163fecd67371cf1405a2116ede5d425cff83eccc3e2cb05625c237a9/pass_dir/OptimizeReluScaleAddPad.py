import torch
import triton
import triton.language as tl

def pattern(in0, in1, in2):
    tmp2 = torch.nn.functional.relu(in2, inplace=False)
    tmp3 = in1 * tmp2
    tmp4 = tmp3 + in0
    tmp5 = torch.nn.functional.pad(tmp4, (0, 1, 0, 1), 'constant', None)
    return tmp5
def replacement_args(in0, in1, in2):
    return (in0, in1, in2)

@triton.jit
def optimized_kernel(
    in2_ptr,
    in1_ptr,
    in0_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    padding_h,
    padding_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (H * W)
    
    x = tl.load(in2_ptr + block_start, mask=mask, other=0.0)
    x = tl.where(x >= 0, x, 0.0)
    x = x * tl.load(in1_ptr)
    x = x + tl.load(in0_ptr)
    
    tl.store(out_ptr + block_start, x, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in0, in1, in2):
    in0 = in0.squeeze(0)
    in1 = in1.squeeze(0)
    B = in2.shape[0]
    C = in2.shape[1]
    H = in2.shape[2]
    W = in2.shape[3]
    padding_h = 1
    padding_w = 1
    
    out = torch.empty((B, C, H + padding_h, W + padding_w),
                      dtype=in2.dtype,
                      device=in2.device)
    
    grid = (1,)
    optimized_kernel[grid](
        in2_ptr=in2,
        in1_ptr=in1,
        in0_ptr=in0,
        out_ptr=out,
        B=B,
        C=C,
        H=H,
        W=W,
        padding_h=padding_h,
        padding_w=padding_w,
        BLOCK_SIZE=128,
    )
    
    return out
def replacement_func():
    return kernel_wrapper