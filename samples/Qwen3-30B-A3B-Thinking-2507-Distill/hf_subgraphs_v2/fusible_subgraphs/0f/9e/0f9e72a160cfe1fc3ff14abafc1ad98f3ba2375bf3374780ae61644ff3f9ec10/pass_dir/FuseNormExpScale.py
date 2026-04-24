import torch
import triton
import triton.language as tl


# Fuse ALL 6 ops: normalize in_1, normalize in_2, exp(in_0)*normalize(in_2)
# Returns all 3 observable values (tmp_6, tmp_4, tmp_2)
def pattern(in_0, in_1, in_2):
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return (tmp_6, tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def full_fuse_kernel(
    in_0_ptr,
    in_1_ptr, out_1_ptr,
    in_2_ptr, out_2_ptr, out_2_scaled_ptr,
    B1, D,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base1 = row * D
    base2 = row * D
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    # normalize in_1
    x1 = tl.load(in_1_ptr + base1 + offsets, mask=mask, other=0.0)
    x1_f32 = x1.to(tl.float32)
    sq1 = tl.sum(x1_f32 * x1_f32, axis=0)
    n1 = tl.sqrt(sq1)
    tl.store(out_1_ptr + base1 + offsets, (x1_f32 / n1).to(x1.dtype), mask=mask)

    # normalize in_2
    x2 = tl.load(in_2_ptr + base2 + offsets, mask=mask, other=0.0)
    x2_f32 = x2.to(tl.float32)
    sq2 = tl.sum(x2_f32 * x2_f32, axis=0)
    n2 = tl.sqrt(sq2)
    tl.store(out_2_ptr + base2 + offsets, (x2_f32 / n2).to(x2.dtype), mask=mask)

    # scale by exp(in_0)
    scale = tl.exp(tl.load(in_0_ptr).to(tl.float32))
    tl.store(out_2_scaled_ptr + base2 + offsets, (scale * (x2_f32 / n2)).to(x2.dtype), mask=mask)


@torch.fx.wrap
def full_fuse(in_0, in_1, in_2):
    D = in_1.shape[-1]
    B1 = in_1.numel() // D
    out_1 = torch.empty_like(in_1)
    out_2 = torch.empty_like(in_2)
    out_2_scaled = torch.empty_like(in_2)
    full_fuse_kernel[(B1,)](
        in_0, in_1, out_1, in_2, out_2, out_2_scaled,
        B1, D,
        BLOCK_SIZE=512,
    )
    return (out_2_scaled, out_2, out_1)


def replacement_func():
    return full_fuse