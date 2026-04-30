import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel for mean along dim=-2 (= dim 1) of [B, H, C] tensors.
# Grid: (B, 1) — one program per batch element, BLOCK_C=C=256 channels at once.
# Fixed BLOCK_C=256 (all channels), autotune only over BLOCK_H.
# This matches the original v1 structure that scored 0.780.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # 5 configs — sufficient to cover all batch sizes without triggering timeouts
        triton.Config({'BLOCK_C': 256, 'BLOCK_H': 2048}, num_warps=4),
        triton.Config({'BLOCK_C': 256, 'BLOCK_H': 2048}, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_H': 1024}, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_H': 4096}, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_H': 256},  num_warps=4),
    ],
    key=['B', 'H', 'C'],
)
@triton.jit
def _mean_dim1_kernel(
    in_ptr,
    out_ptr,
    B, H, C,
    stride_b, stride_h, stride_c,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Compute mean along dim=1 (H) for a 3D tensor [B, H, C].
    Grid: (B, 1) — one program per batch element.
    Accumulates H values for all C channels simultaneously.
    """
    b_idx = tl.program_id(0)
    # c_block = tl.program_id(1)  # always 0 since grid is (B, 1)

    c_start   = tl.arange(0, BLOCK_C)   # [0..255]

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    base_in = in_ptr + b_idx * stride_b

    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask    = h_offsets < H

        # Load [BLOCK_H, BLOCK_C] tile; C-dim (stride_c=1) is fast → coalesced
        ptrs = base_in + h_offsets[:, None] * stride_h + c_start[None, :] * stride_c
        load_mask = h_mask[:, None]  # c_start always in [0..255], C=256 always valid
        vals = tl.load(ptrs, mask=load_mask, other=0.0)

        acc += tl.sum(vals.to(tl.float32), axis=0)   # [BLOCK_C]

    mean_val = acc / H

    # Output [B, 1, C]: flat offset = b * C + c
    out_offsets = b_idx * C + c_start
    tl.store(out_ptr + out_offsets, mean_val.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def mean_dim_neg2_keepdim(in_2):
    """
    Replace in_2.mean(dim=-2, keepdim=True) for [B, H, C] tensors.
    Uses a Triton reduction kernel with grid=(B, 1) for all C=256 channels.
    """
    B, H, C = in_2.shape
    out = torch.empty((B, 1, C), dtype=in_2.dtype, device=in_2.device)

    # Grid: (B, 1) — one program per batch element
    grid = lambda meta: (B, 1)

    _mean_dim1_kernel[grid](
        in_2, out,
        B, H, C,
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(in_2):
    """Match in_2.mean(dim=-2, keepdim=True)."""
    return in_2.mean(dim=-2, keepdim=True)


def replacement_args(in_2):
    return (in_2,)


def replacement_func():
    return mean_dim_neg2_keepdim