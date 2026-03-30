import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = torch.sigmoid(in_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def sigmoid_only_kernel(
    in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # Upcast to fp32: tl.exp requires fp32/fp64; handles bf16/fp16 inputs
    x_f32 = x.to(tl.float32)
    out_f32 = 1.0 / (1.0 + tl.exp(-x_f32))
    tl.store(out_ptr + offsets, out_f32.to(x.dtype), mask=mask)


# Cache grids so we don't recompute them on every call
_grid_cache: dict = {}


@torch.fx.wrap
def triton_sigmoid(in_0):
    n = in_0.numel()
    out = torch.empty_like(in_0)
    BLOCK_SIZE = 1024
    key = n
    if key not in _grid_cache:
        _grid_cache[key] = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    grid = _grid_cache[key]
    sigmoid_only_kernel[grid](in_0, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return triton_sigmoid