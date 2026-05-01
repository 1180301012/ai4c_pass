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
        triton.Config({'BLOCK_COLS': 1024}, num_warps=4),
        triton.Config({'BLOCK_COLS': 1024}, num_warps=8),
        triton.Config({'BLOCK_COLS': 512},  num_warps=4),
        triton.Config({'BLOCK_COLS': 512},  num_warps=2),
    ],
    key=['cols'],
)
@triton.jit
def _l2_normalize_kernel(
    x_ptr, out_ptr,
    rows, cols,
    stride_xr, stride_xc,
    stride_or, stride_oc,
    BLOCK_COLS: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_COLS)
    mask = col_offsets < cols

    x_ptrs = x_ptr + row_idx * stride_xr + col_offsets * stride_xc
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Accumulate in fp32 for numerical stability
    x_f32 = x.to(tl.float32)
    sum_sq = tl.sum(x_f32 * x_f32, axis=0)
    norm = tl.sqrt(sum_sq)

    # Clamp to eps=1e-12 (same as PyTorch normalize default)
    norm_safe = tl.maximum(norm, 1e-12)

    # Normalize, cast back to input dtype
    out = (x_f32 / norm_safe).to(x.dtype)

    out_ptrs = out_ptr + row_idx * stride_or + col_offsets * stride_oc
    tl.store(out_ptrs, out, mask=mask)


@torch.fx.wrap
def triton_fuse_cat_l2_normalize(in_0):
    rows = in_0.shape[0]
    cols = in_0.shape[1]
    out = torch.empty_like(in_0)

    grid = (rows,)

    _l2_normalize_kernel[grid](
        in_0, out,
        rows, cols,
        in_0.stride(0), in_0.stride(1),
        out.stride(0), out.stride(1),
    )

    return out


def replacement_func():
    return triton_fuse_cat_l2_normalize