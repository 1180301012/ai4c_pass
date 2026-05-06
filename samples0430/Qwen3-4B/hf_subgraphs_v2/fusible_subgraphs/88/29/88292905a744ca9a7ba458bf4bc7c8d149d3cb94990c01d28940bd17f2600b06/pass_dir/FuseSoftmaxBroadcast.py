import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(in_0.size(0), -1)
    tmp_2 = tmp_1.view(in_0.size(0), 2, -1, 1, 1)
    tmp_3 = tmp_2 * in_0
    tmp_4 = torch.sum(tmp_3, dim=1)
    return (tmp_4.contiguous(),)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    batch_size,
    block_size,
):
    pid0 = tl.program_id(0)
    block_start = pid0 * block_size
    offsets = tl.arange(0, block_size)
    mask = offsets < batch_size

    in0 = tl.load(in0_ptr + block_start, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + block_start, mask=mask, other=0.0)

    exp_in1 = tl.exp(in1)
    sum_exp = tl.sum(exp_in1, dim=1, keepdim=True)
    softmax = exp_in1 / sum_exp

    tmp = softmax * in0
    out = tl.sum(tmp, dim=1)

    tl.store(out_ptr + block_start, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    batch_size = in_0.size(0)
    out = torch.empty(batch_size, dtype=in_0.dtype, device=in_0.device)
    block_size = 256

    optimized_kernel[(batch_size,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        block_size=block_size,
    )
    return out

def replacement_func():
    return kernel_wrapper