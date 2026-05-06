import torch
import triton
import triton.language as tl


def pattern(y):
    out = y.mean(dim=-2, keepdim=True)
    return out


def replacement_args(y):
    return (y,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 128, 'WARPS': 8}),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 256, 'WARPS': 8}),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 64,  'WARPS': 4}),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 128, 'WARPS': 4}),
        triton.Config({'BLOCK_C': 64,  'BLOCK_S': 256, 'WARPS': 4}),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 32,  'WARPS': 2}),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 64,  'WARPS': 2}),
        triton.Config({'BLOCK_C': 64,  'BLOCK_S': 128, 'WARPS': 4}),
    ],
    key=['C', 'S'],
)
@triton.jit
def _mean_dim1_kernel(
    x_ptr, out_ptr,
    B, C, S,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    Reduces x[B, S, C] along axis 1 (dim S) to produce out[B, 1, C].
    Each program handles one tile of [BLOCK_C, BLOCK_S].
    """
    pid_b  = tl.program_id(0)   # batch index
    pid_c  = tl.program_id(1)   # output-channel tile

    c_start = pid_c * BLOCK_C
    c_offs  = c_start + tl.arange(0, BLOCK_C)  # [BLOCK_C]
    c_mask  = c_offs < C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over the S (sequence / reduction) dimension in blocks
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)  # [BLOCK_S]
        s_mask = s_offs < S

        # Load x[pid_b, s_offs, c_offs]: shape [BLOCK_S, BLOCK_C]
        # x is [B, S, C] contiguous with strides [S*C, C, 1]
        x_ptrs = x_ptr + pid_b * (S * C) + s_offs[:, None] * C + c_offs[None, :]
        x_mask = s_mask[:, None] & c_mask[None, :]
        tile   = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_S, BLOCK_C]

        # Accumulate: sum over S dimension
        acc = acc + tl.sum(tile.to(tl.float32), axis=0)

    # Normalize by S (sequence length) and convert back to input dtype
    out_vals = (acc.to(tile.dtype)
                / S)[:, None]                     # [BLOCK_C, 1]  (broadcast)

    # Output layout: [B, 1, C], stride [C, C, 1]
    out_ptrs = out_ptr + pid_b * C + c_offs
    tl.store(out_ptrs, out_vals, mask=c_mask)


@torch.fx.wrap
def triton_mean_dim_neg2_keepdim(y):
    """Replace x.mean(dim=-2, keepdim=True) with a fast Triton reduction."""
    B, S, C = y.shape
    out = torch.empty((B, 1, C), dtype=y.dtype, device=y.device)
    grid = lambda meta: (B, triton.cdiv(C, meta['BLOCK_C']))

    _mean_dim1_kernel[grid](y, out, B, C, S)
    return out


def replacement_func():
    return triton_mean_dim_neg2_keepdim