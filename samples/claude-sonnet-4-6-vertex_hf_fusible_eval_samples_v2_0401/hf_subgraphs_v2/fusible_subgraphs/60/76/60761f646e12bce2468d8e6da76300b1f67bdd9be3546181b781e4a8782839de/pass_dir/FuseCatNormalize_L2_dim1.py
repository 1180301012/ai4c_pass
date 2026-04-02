import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_0 = torch.cat([x], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return tmp_1


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['D'],
)
@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    # Load row and upcast to float32 for precision
    x = tl.load(x_ptr + row_idx * D + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Compute L2 norm along dim=1 (across the row)
    sq_sum = tl.sum(x_f32 * x_f32, axis=0)
    norm = tl.sqrt(sq_sum)
    # Default eps for torch.nn.functional.normalize is 1e-12
    norm = tl.maximum(norm, 1e-12)

    # Normalize and cast back to original dtype
    out = (x_f32 / norm).to(x.dtype)
    tl.store(out_ptr + row_idx * D + offsets, out, mask=mask)


@torch.fx.wrap
def l2_normalize_wrapper(x):
    B, D = x.shape
    out = torch.empty_like(x)
    # One program per row; autotune selects best BLOCK_SIZE
    grid = (B,)
    l2_normalize_kernel[grid](x, out, D)
    return out


def replacement_func():
    return l2_normalize_wrapper