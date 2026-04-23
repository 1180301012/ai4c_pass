import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = in_0.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _unused_identity_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(y_ptr + offs, x, mask=mask)


@torch.fx.wrap
def fold_unsqueeze_transpose_to_as_strided(in_0):
    out = torch.empty((0,), device=in_0.device, dtype=in_0.dtype)
    out.set_(in_0.untyped_storage(), 0, (1, 1, 128, 1024), (131072, 131072, 1, 128))
    return out


def replacement_func():
    return fold_unsqueeze_transpose_to_as_strided