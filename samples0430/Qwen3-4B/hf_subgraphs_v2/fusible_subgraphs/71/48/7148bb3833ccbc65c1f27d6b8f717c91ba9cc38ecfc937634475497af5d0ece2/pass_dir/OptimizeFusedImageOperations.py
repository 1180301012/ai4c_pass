import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = tmp_2 + tmp_7
    tmp_9 = in_1.unsqueeze(-1)
    tmp_10 = tmp_9.unsqueeze(-1)
    return (tmp_8, tmp_10)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out0_ptr,
    out1_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < tl.min(H, W)
    in2 = tl.load(in2_ptr + off, mask=mask, other=0.0)
    relu_in2 = tl.where(in2 > 0, in2, 0.0)
    pool = tl.zeros(BLOCK_SIZE, dtype=in2.dtype)
    tmp4 = pool - relu_in2
    in0 = tl.load(in0_ptr + off, mask=mask, other=0.0)
    out0 = relu_in2 + in0 * tmp4
    tl.store(out0_ptr + off, out0, mask=mask)

def kernel_wrapper(in_0, in_1, in_2):
    bs, c, h, w = in_2.shape
    out0 = torch.empty((bs, c, h, w), dtype=in_2.dtype, device=in_2.device)
    out1 = torch.empty((c, 1, 1), dtype=in_1.dtype, device=in_1.device)
    grid = (1, 1)
    optimized_kernel[(grid)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        out0_ptr=out0,
        out1_ptr=out1,
        B=bs,
        C=c,
        H=h,
        W=w,
        BLOCK_SIZE=128
    )
    return out0, out1

def replacement_func():
    return kernel_wrapper