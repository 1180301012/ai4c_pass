import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.silu(in_0, inplace = True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_silu_detach(in_0, in_1, in_2):
    N = in_0.numel()
    silu_out = torch.empty_like(in_0)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    silu_kernel[grid](
        x_ptr=in_0,
        out_ptr=silu_out,
        n_elements=N,
    )

    # detach is identity in forward pass, so just return inputs directly
    return (in_1, in_2, silu_out, silu_out)

def replacement_func():
    return fused_silu_detach