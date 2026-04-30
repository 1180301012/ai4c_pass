import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 1024}, num_warps=4),
        triton.Config({'BLOCK_D': 1024}, num_warps=8),
        triton.Config({'BLOCK_D': 2048}, num_warps=4),
        triton.Config({'BLOCK_D': 2048}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def l2_normalize_dim1_kernel(
    x_ptr,
    out_ptr,
    B,
    D,
    stride_b,
    BLOCK_D: tl.constexpr,
):
    """Each program handles one row (one sample). Computes L2 norm along dim=1."""
    b_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    # Load row in fp32 for precision
    x = tl.load(x_ptr + b_idx * stride_b + offsets, mask=mask, other=0.0).to(tl.float32)

    # L2 norm: sqrt(sum(x^2))
    x_sq_sum = tl.sum(x * x, axis=0)
    norm = tl.sqrt(x_sq_sum)

    # Safe division: if norm is 0, return 0 (won't affect masked positions)
    x_norm = x / (norm + 1e-8)

    # Store back in original dtype
    tl.store(out_ptr + b_idx * stride_b + offsets, x_norm.to(x_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_cat_normalize_dim1(in_0):
    B, D = in_0.shape
    out = torch.empty_like(in_0)

    # torch.cat([in_0], dim=1) on a 2D tensor is shape-preserving,
    # so we can skip it and normalize in_0 directly.
    stride_b = in_0.stride(0)

    l2_normalize_dim1_kernel[(B,)](
        in_0,
        out,
        B,
        D,
        stride_b,
    )

    return out


def replacement_func():
    return fused_cat_normalize_dim1