import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels for K_T: permute(0,2,1,3) + transpose(-2,-1)
#
# Input  K  : [B, S, H, DK]  non-contiguous, H-stride = D_FULL=192
# Output K_T: [B, H, DK, S]  contiguous
#
# Key: load the [DK, BLOCK_S] block with transposed index order so stores
# are fully coalesced (K_T[b,h,dk,:] is written contiguously).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 64}, num_warps=2),
        triton.Config({"BLOCK_S": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 64}, num_warps=8),
    ],
    key=["B"],
)
@triton.jit
def _permute_kt_kernel(
    x_ptr,       # [B, S, H, DK]  non-contiguous
    out_ptr,     # [B, H, DK, S]  contiguous
    B, S, H,
    D_FULL: tl.constexpr,   # 192
    DK: tl.constexpr,       # 32
    BLOCK_S: tl.constexpr,  # >= S, power of 2 (e.g. 64)
):
    """
    One program per (b, h).
    Loads  K[b, s, h, dk]   as a [DK, BLOCK_S] block (transposed indices).
    Stores K_T[b, h, dk, s] as a [DK, BLOCK_S] block (contiguous in s).
    """
    pid = tl.program_id(0)
    h = pid % H
    b = pid // H

    s  = tl.arange(0, BLOCK_S)          # [BLOCK_S]
    dk = tl.arange(0, DK)               # [DK]
    s_mask = s < S

    # Load K[b, s, h, dk]: address = b*(S*H*D_FULL) + s*(H*D_FULL) + h*D_FULL + dk
    # Arrange as [DK, BLOCK_S] so stores are contiguous
    in_base = b * (S * H * D_FULL) + h * D_FULL
    in_offsets = in_base + s[None, :] * (H * D_FULL) + dk[:, None]  # [DK, BLOCK_S]
    block = tl.load(x_ptr + in_offsets, mask=s_mask[None, :], other=0.0)

    # Store K_T[b, h, dk, s]: address = b*(H*DK*S) + h*(DK*S) + dk*S + s
    # Contiguous in s (last dim of K_T)
    out_base = b * (H * DK * S) + h * (DK * S)
    out_offsets = out_base + dk[:, None] * S + s[None, :]  # [DK, BLOCK_S]
    tl.store(out_ptr + out_offsets, block, mask=s_mask[None, :])


def _triton_permute_kt(x):
    """x: [B,S,H,32] non-contiguous → K_T [B,H,32,S] contiguous"""
    B, S, H = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    out = torch.empty((B, H, 32, S), dtype=x.dtype, device=x.device)
    _permute_kt_kernel[(B * H,)](
        x, out, B, S, H,
        D_FULL=192, DK=32,
    )
    return out


# ---------------------------------------------------------------------------
# Top-level @torch.fx.wrap dispatch — must be at module level
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _dispatch_qkv(x, route):
    return _triton_permute_kt(x)


# ---------------------------------------------------------------------------
# Pattern: permute(0,2,1,3) + transpose(-2,-1)  →  K_T  [SINGLE OUTPUT]
# ---------------------------------------------------------------------------

def pattern(k_in):
    t   = k_in.permute(0, 2, 1, 3)
    out = t.transpose(-2, -1)
    return out


def replacement_args(k_in):
    return (k_in, "kt")


def replacement_func():
    return _dispatch_qkv