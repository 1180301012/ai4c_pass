import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


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
def fused_silu_mul_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    # Compute silu(x) * y = x * sigmoid(x) * y
    sigmoid_x = tl.sigmoid(x.to(tl.float32))
    silu_x = x * sigmoid_x.to(x.dtype)
    result = silu_x * y

    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_silu_mul(in_0, in_1):
    N = in_0.numel()
    out = torch.empty_like(in_0)

    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_silu_mul_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        n_elements=N,
    )

    return (out,)


def replacement_func():
    return fused_silu_mul