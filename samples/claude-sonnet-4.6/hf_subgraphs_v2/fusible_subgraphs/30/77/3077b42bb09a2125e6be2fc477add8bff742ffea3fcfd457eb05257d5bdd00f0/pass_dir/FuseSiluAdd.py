import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_1)
    tmp_1 = tmp_0 + in_0
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_silu_add_kernel(
    x_ptr,      # in_1: SiLU input
    y_ptr,      # in_0: residual add input
    out_ptr,    # output
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # SiLU: x * sigmoid(x), upcast to fp32 for numerical precision
    x_f32 = x.to(tl.float32)
    silu_x = (x_f32 * tl.sigmoid(x_f32)).to(x.dtype)

    out = silu_x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_add(in_0, in_1):
    N = in_0.numel()
    out = torch.empty_like(in_0)

    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_silu_add_kernel[grid](
        in_1,
        in_0,
        out,
        N,
    )

    return out


def replacement_func():
    return fused_silu_add