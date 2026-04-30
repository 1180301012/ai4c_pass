import torch
import triton
import triton.language as tl


def pattern(x):
    sig = x.sigmoid()
    sub = sig - 0.25
    mul = sub * 3.141592653589793
    return mul


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_sigmoid_sub_mul_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values and upcast to float32 for accuracy
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute (sigmoid(x) - 0.25) * pi
    sig = tl.sigmoid(x)
    result = (sig - 0.25) * 3.141592653589793

    # Store result (Triton handles dtype conversion automatically)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_sigmoid_sub_mul(x):
    n_elements = x.numel()
    if n_elements == 0:
        return torch.empty_like(x)

    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_sigmoid_sub_mul_kernel[grid](
        x_ptr=x, out_ptr=out, n_elements=n_elements,
    )

    return out


def replacement_func():
    return fused_sigmoid_sub_mul