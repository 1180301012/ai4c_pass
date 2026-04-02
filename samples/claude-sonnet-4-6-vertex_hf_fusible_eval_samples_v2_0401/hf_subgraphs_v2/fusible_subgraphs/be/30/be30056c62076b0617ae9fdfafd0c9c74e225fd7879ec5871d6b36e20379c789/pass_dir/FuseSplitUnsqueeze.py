import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: split (leaf "split") + getitems + aten.unsqueeze.default
# Confirmed working from diagnostics. The silu predecessor is not matched,
# so our replacement receives the already-silu'd tensor and writes it to
# three separate CONTIGUOUS output buffers (the split views are non-contiguous).
# ---------------------------------------------------------------------------
def pattern(in_1_silu):
    split = torch.functional.split(in_1_silu, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)   # method form — confirmed matching in diagnostics
    return (tmp_3, tmp_6, tmp_4)


def replacement_args(in_1_silu):
    return (in_1_silu,)


# ---------------------------------------------------------------------------
# Triton kernel: one program per (batch, seq) row.
# Reads C=1152 elements, writes to three contiguous output buffers.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 2048}, num_warps=4),
        triton.Config({'BLOCK_C': 2048}, num_warps=8),
        triton.Config({'BLOCK_C': 2048}, num_warps=16),
        triton.Config({'BLOCK_C': 2048}, num_warps=2),
    ],
    key=['N_ROWS'],
)
@triton.jit
def _split_kernel(
    in_ptr,
    out_u_ptr,
    out_v_ptr,
    out_g_ptr,
    N_ROWS,
    C:   tl.constexpr,   # 1152
    C_U: tl.constexpr,   #  512
    C_V: tl.constexpr,   #  512
    C_G: tl.constexpr,   #  128
    BLOCK_C: tl.constexpr,  # 2048
):
    pid = tl.program_id(0)
    c_off = tl.arange(0, BLOCK_C)

    mask_load = c_off < C
    x = tl.load(in_ptr + pid * C + c_off, mask=mask_load, other=0.0)

    u_mask = c_off < C_U
    tl.store(out_u_ptr + pid * C_U + c_off, x, mask=u_mask)

    v_mask = (c_off >= C_U) & (c_off < C_U + C_V)
    tl.store(out_v_ptr + pid * C_V + (c_off - C_U), x, mask=v_mask)

    g_mask = c_off >= C_U + C_V
    tl.store(out_g_ptr + pid * C_G + (c_off - C_U - C_V), x, mask=g_mask)


@torch.fx.wrap
def split_unsqueeze_triton(in_1_silu):
    B, S, C = in_1_silu.shape
    C_U, C_V, C_G = 512, 512, 128
    N_ROWS = B * S

    out_u    = torch.empty(B, S, C_U, dtype=in_1_silu.dtype, device=in_1_silu.device)
    out_v    = torch.empty(B, S, C_V, dtype=in_1_silu.dtype, device=in_1_silu.device)
    out_g_3d = torch.empty(B, S, C_G, dtype=in_1_silu.dtype, device=in_1_silu.device)

    _split_kernel[(N_ROWS,)](
        in_1_silu, out_u, out_v, out_g_3d,
        N_ROWS,
        C, C_U, C_V, C_G,
    )

    out_g = out_g_3d.unsqueeze(2)          # [B, S, 1, C_G]
    return (out_u, out_g, out_v)


def replacement_func():
    return split_unsqueeze_triton