import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror model.py exactly in op structure and observable outputs.
def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return (tmp_7, tmp_3, tmp_6, tmp_4)


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit

def silu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)
    x_f32 = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x_f32))
    y = x_f32 * sig
    tl.store(y_ptr + offs, y.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_silu_split_unsqueeze_gamma_view(in_0, in_1):
    # Compute SiLU into a fresh contiguous output buffer.
    tmp_1 = torch.empty_like(in_1)
    n_elements = in_1.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](
        x_ptr=in_1,
        y_ptr=tmp_1,
        n_elements=n_elements,
    )

    # Recreate the exact observable outputs as cheap metadata-only views.
    tmp_3 = tmp_1[:, :, :512]
    tmp_4 = tmp_1[:, :, 512:1024]
    tmp_6 = tmp_1[:, :, 1024:1152].unsqueeze(2)
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return (tmp_7, tmp_3, tmp_6, tmp_4)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_silu_split_unsqueeze_gamma_view