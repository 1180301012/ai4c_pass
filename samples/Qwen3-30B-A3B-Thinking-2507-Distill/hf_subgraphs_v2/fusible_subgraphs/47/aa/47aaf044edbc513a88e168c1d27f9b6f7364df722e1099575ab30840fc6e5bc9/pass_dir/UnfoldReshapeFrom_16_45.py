import torch
import triton
import triton.language as tl


# Pattern: transpose + reshape + reshape (starting from tmp_2 after unfold)
# tmp_2: [1, 45, 16, 9] (contiguous, unfold output)
# This matches ALL 3 tiny-model graphs (float16, bfloat16, float32)
def pattern(tmp_2):
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(tmp_2):
    return (tmp_2,)


@triton.jit
def _tiny_reshape_kernel(
    x_ptr, out_ptr,
    L, C, H, total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    hw = offsets % (H * 9)
    n = offsets // (H * 9)
    h = hw // 9
    w = hw % 9
    p = h + n * H
    c = n % C
    row = p + w - 4
    safe_row = tl.maximum(tl.minimum(row, L - 1), 0)
    # tmp_2[0, p, c, w] = x_ptr + p*(C*9) + c*9 + w
    x_val = tl.load(x_ptr + p * (C * 9) + c * 9 + w, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x_val, mask=mask)


@torch.fx.wrap
def _tiny_reshape_16_45(tmp_2):
    L, C, H = 45, 16, 8
    total = 90 * 8 * 9
    out = torch.empty((90, 8, 9), dtype=tmp_2.dtype, device=tmp_2.device)
    _tiny_reshape_kernel[(7,)](tmp_2, out, L, C, H, total, BLOCK_SIZE=1024)
    return out


def replacement_func():
    return _tiny_reshape_16_45