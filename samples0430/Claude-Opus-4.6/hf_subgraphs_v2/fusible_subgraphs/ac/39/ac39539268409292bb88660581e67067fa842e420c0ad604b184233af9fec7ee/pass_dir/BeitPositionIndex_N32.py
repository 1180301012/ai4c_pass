import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    return tmp_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def cat_kernel(
    in1_ptr, in0_ptr, out_ptr,
    split_point,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    is_in1 = offsets < split_point
    in1_val = tl.load(in1_ptr + offsets, mask=is_in1, other=0.0)
    in0_val = tl.load(in0_ptr + (offsets - split_point), mask=(~is_in1) & mask, other=0.0)
    val = tl.where(is_in1, in1_val, in0_val)
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def triton_cat(in_0, in_1):
    M = in_1.shape[0]
    K = in_0.shape[0]
    C = in_1.shape[1]
    total = (M + K) * C
    split_point = M * C
    out = torch.empty((M + K, C), dtype=in_1.dtype, device=in_1.device)
    BLOCK_SIZE = 1024
    num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    cat_kernel[(num_blocks,)](
        in1_ptr=in_1, in0_ptr=in_0, out_ptr=out,
        split_point=split_point, total=total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return triton_cat