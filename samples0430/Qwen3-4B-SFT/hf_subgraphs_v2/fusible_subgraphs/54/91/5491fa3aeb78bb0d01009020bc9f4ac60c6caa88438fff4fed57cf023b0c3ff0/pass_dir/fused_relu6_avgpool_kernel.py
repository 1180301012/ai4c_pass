"""
Shared Triton kernel for fused ReLU6 + global average pooling.
Applied as one kernel to eliminate the intermediate hardtanh buffer.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_N >= 2 so that tl.zeros([BLOCK_N]) always matches sum(x, axis=1) output shape
        # BLOCK_HW must be >= max(HW) to run in a single loop iteration (max HW = 256 / 16x16)
        triton.Config({'BLOCK_N': 2,  'BLOCK_HW': 256}, num_warps=2),
        triton.Config({'BLOCK_N': 4,  'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 8,  'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 16, 'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 32, 'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 2,  'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 4,  'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 8,  'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 4,  'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_N': 8,  'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_N': 16, 'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_N': 4,  'BLOCK_HW': 32},  num_warps=2),
        triton.Config({'BLOCK_N': 8,  'BLOCK_HW': 32},  num_warps=2),
    ],
    key=['HW'],
)
@triton.jit
def relu6_gavgpool_kernel(
    x_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_N:  tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    2-D tiled reduction: each program handles BLOCK_N rows, each of length HW.
    Grid: (ceil(B*C / BLOCK_N),)
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N
    offsets_n  = tl.arange(0, BLOCK_N)            # [BLOCK_N]

    # Accumulator: shape [BLOCK_N]  (sum over HW axis)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for i_hw in range(0, tl.cdiv(HW, BLOCK_HW)):
        hw_start   = i_hw * BLOCK_HW
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)   # [BLOCK_HW]
        hw_mask    = hw_offsets < HW                      # [BLOCK_HW]

        # Build 2-D pointer tensor [BLOCK_N, BLOCK_HW]:
        #   row_idx [:,None] → [BLOCK_N, 1]
        #   col_idx [None,:] → [1,      BLOCK_HW]
        # Both are initialized with tl.zeros + tl.arange,
        # avoiding any implicit broadcast that Triton's SSA
        # shape-inference can't statically resolve.
        row_idx = (row_start + offsets_n)[:, None] * HW   # [BLOCK_N,    1  ]
        col_idx = hw_offsets[None, :]                      # [1, BLOCK_HW]
        row_ptrs = row_idx + col_idx                        # [BLOCK_N, BLOCK_HW]

        mask2d = hw_mask[None, :]                           # [1, BLOCK_HW]  → broadcast to [BLOCK_N, BLOCK_HW]

        x = tl.load(x_ptr + row_ptrs, mask=mask2d, other=0.0)
        x = tl.where(x < 0.0, 0.0, x)      # ReLU
        x = tl.where(x > 6.0, 6.0, x)      # clamp to 6
        acc += tl.sum(x.to(tl.float32), axis=1)    # [BLOCK_N]

    avg  = acc / HW
    tl.store(out_ptr + row_start + offsets_n, avg.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def _call_relu6_gavgpool(in_0, B, C):
    HW  = in_0.shape[-1] * in_0.shape[-2]
    N_R = B * C
    out = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    def grid(meta):
        return (triton.cdiv(N_R, meta['BLOCK_N']),)

    relu6_gavgpool_kernel[grid](x_ptr=in_0, out_ptr=out, C=C, HW=HW)
    return out


@torch.fx.wrap
def _shared_relu6_gavgpool_dispatch(in_0, route):
    """
    Single dispatch wrapper shared by all B-specific passes.
    route encodes which (B, C) config to use.
    C is always 1280 across all graphs being optimized.
    """
    if route == "B1":
        return _call_relu6_gavgpool(in_0, 1, 1280)
    elif route == "B2":
        return _call_relu6_gavgpool(in_0, 2, 1280)
    elif route == "B4":
        return _call_relu6_gavgpool(in_0, 4, 1280)
    elif route == "B8":
        return _call_relu6_gavgpool(in_0, 8, 1280)
    elif route == "B32":
        return _call_relu6_gavgpool(in_0, 32, 1280)
    else:                                  # "B64"
        return _call_relu6_gavgpool(in_0, 64, 1280)