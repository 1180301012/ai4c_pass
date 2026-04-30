import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 64}, num_warps=8),
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=16),
    ],
    key=['N_ROWS', 'D'],
)
@triton.jit
def _fused_norm_kernel(
    in1_ptr,
    in0_ptr,
    out_ptr,
    norm_ptr,
    N_ROWS,
    D,
    MIN_VAL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_D)
    mask = col_offsets < D

    row_start = row_idx * D

    x_raw = tl.load(in1_ptr + row_start + col_offsets, mask=mask, other=0.0)
    w_val = tl.load(in0_ptr)
    norm_val = tl.load(norm_ptr + row_idx)

    x_f32 = x_raw.to(tl.float32)
    w_f32 = w_val.to(tl.float32)
    norm_f32 = norm_val.to(tl.float32)

    clamped_norm = tl.maximum(norm_f32 * 0.07216878364870322, MIN_VAL)
    out_f32 = (x_f32 / clamped_norm) * w_f32

    tl.store(out_ptr + row_start + col_offsets, out_f32.to(x_raw.dtype), mask=mask)


def _run_fused_norm(in_0, norm_out, flat, scale, min_val):
    D = flat.shape[-1]
    N_ROWS = flat.numel() // D
    out = torch.empty_like(flat)
    _fused_norm_kernel[(N_ROWS,)](
        in1_ptr=flat,
        in0_ptr=in_0,
        out_ptr=out,
        norm_ptr=norm_out,
        N_ROWS=N_ROWS,
        D=D,
        MIN_VAL=min_val,
    )
    return out


def pattern(in_0, norm_out, flat):
    tmp_4 = norm_out * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = flat / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7


def replacement_args(in_0, norm_out, flat):
    return (in_0, norm_out, flat, "route_007216878364870322")


@torch.fx.wrap
def _fused_norm_dispatch_007216878364870322(in_0, norm_out, flat, route):
    if route == "route_014433756729740643":
        return _run_fused_norm(in_0, norm_out, flat, 0.14433756729740643, 1e-05)
    elif route == "route_007216878364870322":
        return _run_fused_norm(in_0, norm_out, flat, 0.07216878364870322, 1e-05)
    else:
        return _run_fused_norm(in_0, norm_out, flat, 0.07216878364870322, 1e-05)


def replacement_func():
    return _fused_norm_dispatch_007216878364870322