import torch
import triton
import triton.language as tl


def pattern(in_3, in_4):
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4


def replacement_args(in_3, in_4):
    return (in_3, in_4)


# Shared element-wise add body: load x, load y, store x+y.
# evict_first: streaming hint — data is not reused, avoids L2 pollution.
@triton.jit
def _fused_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    tl.store(out_ptr + offsets, x + y, mask=mask, eviction_policy='evict_first')


# Autotuned variant for large tensors (≥4M elements).
# key=n_elements → one optimal config per problem size.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_add_kernel_large(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    tl.store(out_ptr + offsets, x + y, mask=mask, eviction_policy='evict_first')


@torch.fx.wrap
def fused_add_dropout2d_inference(in_3, in_4):
    out = torch.empty_like(in_3)
    N = in_3.numel()

    # For small tensors (float32/batch=1 → N=2M), the autotune overhead
    # pollutes timing.  Use a fixed, well-tuned config instead.
    if N <= 4194304:  # ≤ 4M elements
        BLOCK_SIZE = 4096
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _fused_add_kernel[grid](in_3, in_4, out, N,
                                BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    else:
        grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
        _fused_add_kernel_large[grid](in_3, in_4, out, N)

    return out


def replacement_func():
    return fused_add_dropout2d_inference