import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_7 = torch.reshape(in_1, [1, -1, in_1.shape[1] // 64, 64])
    return tmp_7


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def _copy_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


@torch.fx.wrap
def reshape_conv_out(in_1):
    s0 = in_1.shape[0]
    s1 = in_1.shape[1]
    out = torch.empty((1, s0, s1 // 64, 64), device=in_1.device, dtype=in_1.dtype)
    n = in_1.numel()
    grid = ((n + 255) // 256,)
    _copy_kernel[grid](in_1, out, n, BLOCK=256)
    return out


def replacement_func():
    return reshape_conv_out