import torch
import triton
import triton.language as tl


# Pattern: torch.sigmoid is the non-inplace form dynamo records after
# functionalizing relu.  Only sigmoid matched in all previous attempts.
def pattern(in_0):
    tmp_1 = torch.sigmoid(in_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute in fp32 for bf16/fp16 correctness
    x_f32 = x.to(tl.float32)
    out_f32 = 1.0 / (1.0 + tl.exp(-x_f32))
    out = out_f32.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_sigmoid(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _sigmoid_kernel[grid](in_0, out, n_elements)
    return out


def replacement_func():
    return triton_sigmoid