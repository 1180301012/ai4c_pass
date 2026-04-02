import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: high-level silu WITHOUT inplace (inplace=True causes proxy aliasing
# in the pattern tracer, making split incorrectly consume in_1 directly).
# Without inplace, silu creates a NEW proxy tmp_1, giving the correct graph.
# split (leaf) + method unsqueeze confirmed from diagnostics.
# ---------------------------------------------------------------------------
def pattern(in_1):
    tmp_1 = torch.nn.functional.silu(in_1)   # no inplace — avoids proxy aliasing
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    return (tmp_3, tmp_6, tmp_4)


def replacement_args(in_1):
    return (in_1,)


# ---------------------------------------------------------------------------
# Triton kernel: reads one row of C=1152 elements, applies SiLU, then writes
# to three separate contiguous output buffers of widths C_U, C_V, C_G.
# One Triton program = one (batch, seq) row.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 2048}, num_warps=2),
        triton.Config({'BLOCK_C': 2048}, num_warps=4),
        triton.Config({'BLOCK_C': 2048}, num_warps=8),
        triton.Config({'BLOCK_C': 2048}, num_warps=16),
    ],
    key=['N_ROWS'],
)
@triton.jit
def _silu_split_kernel(
    in_ptr,
    out_u_ptr,
    out_v_ptr,
    out_g_ptr,
    N_ROWS,
    C:    tl.constexpr,   # total channels  = 1152
    C_U:  tl.constexpr,   # first  split    =  512
    C_V:  tl.constexpr,   # second split    =  512
    C_G:  tl.constexpr,   # third  split    =  128
    BLOCK_C: tl.constexpr, # >= C, power-of-2 = 2048
):
    pid = tl.program_id(0)

    c_off = tl.arange(0, BLOCK_C)

    # ------------------------------------------------------------------
    # Load one row of C elements (mask the padding beyond C)
    # ------------------------------------------------------------------
    ld_mask = c_off < C
    x = tl.load(in_ptr + pid * C + c_off, mask=ld_mask, other=0.0)

    # ------------------------------------------------------------------
    # SiLU in float32 for numerical accuracy, cast back to input dtype
    # ------------------------------------------------------------------
    x_fp32  = x.to(tl.float32)
    silu_x  = (x_fp32 * tl.sigmoid(x_fp32)).to(x.dtype)

    # ------------------------------------------------------------------
    # Write U  [0, C_U)
    # ------------------------------------------------------------------
    u_mask = c_off < C_U
    tl.store(out_u_ptr + pid * C_U + c_off, silu_x, mask=u_mask)

    # ------------------------------------------------------------------
    # Write V  [C_U, C_U+C_V)
    # ------------------------------------------------------------------
    v_mask = (c_off >= C_U) & (c_off < C_U + C_V)
    tl.store(out_v_ptr + pid * C_V + (c_off - C_U), silu_x, mask=v_mask)

    # ------------------------------------------------------------------
    # Write G  [C_U+C_V, C)
    # ------------------------------------------------------------------
    g_mask = c_off >= C_U + C_V
    tl.store(out_g_ptr + pid * C_G + (c_off - C_U - C_V), silu_x, mask=g_mask)


# ---------------------------------------------------------------------------
# Wrapper called by the replacement – must be decorated with @torch.fx.wrap
# ---------------------------------------------------------------------------
@torch.fx.wrap
def silu_split_unsqueeze_kernel_wrapper(in_1):
    B, S, C = in_1.shape          # e.g. [512, 17, 1152]
    C_U, C_V, C_G = 512, 512, 128
    N_ROWS = B * S

    out_u    = torch.empty(B, S, C_U, dtype=in_1.dtype, device=in_1.device)
    out_v    = torch.empty(B, S, C_V, dtype=in_1.dtype, device=in_1.device)
    out_g_3d = torch.empty(B, S, C_G, dtype=in_1.dtype, device=in_1.device)

    _silu_split_kernel[(N_ROWS,)](
        in_1, out_u, out_v, out_g_3d,
        N_ROWS,
        C, C_U, C_V, C_G,
    )

    # unsqueeze(2): [B, S, C_G] -> [B, S, 1, C_G]
    out_g = out_g_3d.unsqueeze(2)

    # Return order must match pattern: (tmp_3, tmp_6, tmp_4)
    return (out_u, out_g, out_v)


def replacement_func():
    return silu_split_unsqueeze_kernel_wrapper