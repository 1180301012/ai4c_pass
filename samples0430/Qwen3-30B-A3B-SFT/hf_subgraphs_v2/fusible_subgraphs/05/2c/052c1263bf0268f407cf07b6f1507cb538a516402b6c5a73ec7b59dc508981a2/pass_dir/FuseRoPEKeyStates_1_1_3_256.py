import torch
import triton
import triton.language as tl


# Pattern: RoPE for key states (model.py lines 6-15 only).
# tmp_6 is observable because it appears in the model's return.
def pattern(in_2, in_1, in_4):
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    return tmp_6


def replacement_args(in_2, in_1, in_4):
    return (in_2, in_1, in_4)


# Kernel: B=1 H=1 S=3 D=256, all params constexpr, single program.
@triton.jit
def rope_key_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    D: tl.constexpr,
    H_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    d_off = tl.arange(0, BLOCK_D)
    for s in range(3):
        base = s * D
        x   = tl.load(x_ptr   + base + d_off)
        cos = tl.load(cos_ptr + base + d_off)
        sin = tl.load(sin_ptr + base + d_off)
        # rotate_x[d<128] = -x[d+128];  rotate_x[d>=128] = x[d-128]
        upper = tl.where(d_off < H_DIM, d_off + H_DIM, d_off)
        lower = tl.where(d_off >= H_DIM, d_off - H_DIM, d_off)
        x_up  = tl.load(x_ptr + base + upper)
        x_low = tl.load(x_ptr + base + lower)
        rot   = tl.where(d_off < H_DIM, -x_up, x_low)
        tl.store(out_ptr + base + d_off, x * cos + rot * sin)


@torch.fx.wrap
def rope_key_func(in_2, in_1, in_4):
    out = torch.empty_like(in_2)
    rope_key_kernel[(1,)](
        in_2, in_1, in_4, out,
        D=256, H_DIM=128, BLOCK_D=256,
        num_warps=4,
    )
    return out


def replacement_func():
    return rope_key_func