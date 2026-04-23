import torch
import triton
import triton.language as tl


def pattern(in_5):
    tmp_10 = in_5[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    return tmp_12


def replacement_args(in_5):
    return (in_5, "expand_repeat")


@triton.jit
def rotary_embed_kernel(
    key_ptr, cos_ptr, sin_ptr, out_ptr,
    n_elements, n_cols, half_dim,
):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < n_elements

    key = tl.load(key_ptr + offsets, mask=mask, other=0.0)
    cos_val = tl.load(cos_ptr + offsets, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptr + offsets, mask=mask, other=0.0)

    col_idx = offsets % n_cols
    paired_offsets = tl.where(col_idx < half_dim, offsets + half_dim, offsets - half_dim)
    paired_key = tl.load(key_ptr + paired_offsets, mask=mask, other=0.0)

    rotated = tl.where(col_idx < half_dim, -paired_key, paired_key)
    out = key * cos_val + rotated * sin_val

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def _rotary_embed_impl(cos, key, sin):
    n_elements = key.numel()
    n_cols = key.shape[-1]
    half_dim = n_cols // 2

    out = torch.empty_like(key)

    rotary_embed_kernel[(1,)](
        key_ptr=key,
        cos_ptr=cos,
        sin_ptr=sin,
        out_ptr=out,
        n_elements=n_elements,
        n_cols=n_cols,
        half_dim=half_dim,
    )

    return out


@triton.jit
def expand_repeat_kernel(
    in_ptr, out_ptr,
    n_heads: tl.constexpr,
    n_seq: tl.constexpr,
    full_dim: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid // n_seq
    s = pid % n_seq

    src_row_start = s * full_dim
    src_offsets = src_row_start + tl.arange(0, full_dim)

    dst_row_start = (h * n_seq + s) * full_dim
    dst_offsets = dst_row_start + tl.arange(0, full_dim)

    val = tl.load(in_ptr + src_offsets)
    tl.store(out_ptr + dst_offsets, val)


@torch.fx.wrap
def _expand_repeat_impl(in_5):
    n_seq = 3
    full_dim = 256
    n_heads = 8
    total_rows = n_heads * n_seq

    out = torch.empty((1, n_heads, n_seq, full_dim), dtype=in_5.dtype, device=in_5.device)

    expand_repeat_kernel[(total_rows,)](
        in_ptr=in_5,
        out_ptr=out,
        n_heads=n_heads,
        n_seq=n_seq,
        full_dim=full_dim,
    )

    return out


# Shared dispatch wrapper used by both passes
@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "rotary_embed":
        return _rotary_embed_impl(args[0], args[1], args[2])
    elif route == "expand_repeat":
        return _expand_repeat_impl(args[0])
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return dispatch_wrapper