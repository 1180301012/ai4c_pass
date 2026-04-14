"""Shared Triton kernel and dispatch wrapper used by both pass files."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def _unfold_scatter_shared(
    in_ptr, out_ptr,
    C9, S, C, H,
    stride_c9, stride_s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Rearranges unfold output [1,C*9,S] -> [S*(C//H), H, 9]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    k = offs % 9
    tmp = offs // 9
    j = tmp % H
    i = tmp // H

    idx = i * H + j
    s   = idx // C
    c   = idx % C

    # Access unfold output: in_0[0, c*9+k, s]
    in_off = (c * 9 + k) * stride_c9 + s * stride_s
    val = tl.load(in_ptr + in_off, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


@torch.fx.wrap
def unfold_window_dispatch(in_0, route):
    """Dispatch wrapper: route='c16_h8' or 'c384_h64'."""
    C9 = in_0.shape[1]          # C * 9
    S  = in_0.shape[2]          # sequence length
    C  = C9 // 9                # number of channels
    H  = 8 if route == "c16_h8" else 64
    n_elements = S * C * 9

    out = torch.empty((S * C // H, H, 9), dtype=in_0.dtype, device=in_0.device)

    stride_c9 = in_0.stride(1)
    stride_s  = in_0.stride(2)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _unfold_scatter_shared[grid](
        in_ptr=in_0, out_ptr=out,
        C9=C9, S=S, C=C, H=H,
        stride_c9=stride_c9, stride_s=stride_s,
        n_elements=n_elements,
    )
    return out