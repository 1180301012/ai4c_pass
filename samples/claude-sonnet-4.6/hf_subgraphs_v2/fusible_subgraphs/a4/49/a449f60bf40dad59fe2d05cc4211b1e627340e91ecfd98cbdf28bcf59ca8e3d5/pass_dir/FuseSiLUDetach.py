import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    return tmp_0


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def triton_silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute SiLU in float32 for precision: x * sigmoid(x)
    x_f32 = x.to(tl.float32)
    sigmoid_x = tl.sigmoid(x_f32)
    out = (x_f32 * sigmoid_x).to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_silu(in_0):
    n = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    triton_silu_kernel[grid](in_0, out, n)
    return out


def replacement_func():
    return triton_silu