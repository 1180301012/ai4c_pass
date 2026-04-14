import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: in_2[:, :, 1:, :].transpose(-1, -2)
#
# All target graphs share this structure: slice along dim-2 then transpose
# the last two dims.  The output (tmp_3) feeds into reshape→split which
# both become zero-copy views once tmp_3 is contiguous.
# ---------------------------------------------------------------------------

def pattern(v):
    tmp_2 = v[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3


def replacement_args(v):
    return (v,)


# ---------------------------------------------------------------------------
# Triton kernel: fused slice-transpose
#
# Each program handles one (bh, d_tile, n_tile) 2-D tile:
#   Load  v[bh, n+1, d]  → [BLOCK_N × BLOCK_D]  (coalesced along d,  stride=1)
#   Store out[bh, d,  n] → [BLOCK_D × BLOCK_N]  (coalesced along n,  stride=1)
#
# No autotune: the kernel is compiled for the selected constexpr sizes.
# Block sizes are passed explicitly at call time from the wrapper.
# ---------------------------------------------------------------------------

@triton.jit
def _slice_transpose_kernel(
    v_ptr, out_ptr,
    BH, N_in, D, N_out,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_out_bh, stride_out_d, stride_out_n,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)   # flattened B*H index
    pid_d  = tl.program_id(1)   # tile index over D
    pid_n  = tl.program_id(2)   # tile index over N_out

    n_start = pid_n * BLOCK_N
    d_start = pid_d * BLOCK_D

    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    d_offs = d_start + tl.arange(0, BLOCK_D)   # [BLOCK_D]

    # Load v[bh, n+1, d]  — d is inner (stride 1) → coalesced
    v_idx = (pid_bh * stride_v_bh
             + (n_offs + 1)[:, None] * stride_v_n
             + d_offs[None, :] * stride_v_d)
    v_mask = (n_offs < N_out)[:, None] & (d_offs < D)[None, :]
    v_data = tl.load(v_ptr + v_idx, mask=v_mask, other=0.0)   # [BLOCK_N, BLOCK_D]

    # Store out[bh, d, n]  — n is inner (stride 1) → coalesced
    out_idx = (pid_bh * stride_out_bh
               + d_offs[:, None] * stride_out_d
               + n_offs[None, :] * stride_out_n)
    out_mask = (d_offs < D)[:, None] & (n_offs < N_out)[None, :]
    tl.store(out_ptr + out_idx, tl.trans(v_data), mask=out_mask)


# ---------------------------------------------------------------------------
# Wrapper — @torch.fx.wrap keeps it opaque to FX tracing.
# Block sizes are chosen statically to cover all target shapes without
# incurring per-call autotune overhead.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def slice_transpose(v):
    """
    Fused slice-then-transpose:
        in  v  : [B, H, N,   D]
        out    : [B, H, D, N-1]  (contiguous)
    Equivalent to  v[:, :, 1:, :].transpose(-1, -2).contiguous()
    """
    B, H, N_in, D = v.shape
    N_out = N_in - 1
    BH    = B * H
    out   = torch.empty((B, H, D, N_out), dtype=v.dtype, device=v.device)

    # Use a 32×32 tile — works for all D and N_out values in the target
    # graphs (D ∈ {16,19,27,32,40,64}, N_out ∈ {49,144,196,784,2304,3136}).
    BLOCK_N = 32
    BLOCK_D = 32

    grid = (
        BH,
        triton.cdiv(D,     BLOCK_D),
        triton.cdiv(N_out, BLOCK_N),
    )

    _slice_transpose_kernel[grid](
        v, out,
        BH, N_in, D, N_out,
        v.stride(1),   v.stride(2),   v.stride(3),
        out.stride(1), out.stride(2), out.stride(3),
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return out


def replacement_func():
    return slice_transpose