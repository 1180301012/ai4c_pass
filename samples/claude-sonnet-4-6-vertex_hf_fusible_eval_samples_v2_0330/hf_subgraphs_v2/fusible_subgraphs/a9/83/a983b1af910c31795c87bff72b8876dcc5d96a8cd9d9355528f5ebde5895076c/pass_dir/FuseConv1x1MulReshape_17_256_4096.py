"""
Fused pass: 1x1 conv2d + multiply-by-1.0 + reshape(-1, 17, 4096)

Pattern:
    conv2d = torch.conv2d(in_2, in_1, in_0, (1,1), (0,0), (1,1), 1)
    tmp_3  = conv2d * 1.0
    tmp_4  = tmp_3.reshape(-1, 17, 4096)
    return (tmp_4,)

Key shapes (fixed across all variants):
    in_0  : bias   [17]
    in_1  : weight [17, 256, 1, 1]
    in_2  : input  [B, 256, 64, 64]   (B varies: 1, 4, 8, 32, 64, 512)

Optimization:
    - 1x1 conv ≡ matmul: weight[OC, IC] @ input[IC, HW]  +  bias[OC]
    - multiply by 1.0 is elided (no-op)
    - reshape is baked into the output allocation shape
    - Triton kernel with autotuning over BLOCK_N and BLOCK_K
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – must mirror model.py exactly (positional args, same ops)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # 2-D grid: dim0 (x) = B*HW tiles, dim1 (y) = OC tiles
        # All use BLOCK_K=256 (=IC) so the K-loop runs once – zero loop overhead.
        # Vary BLOCK_N and num_warps; autotuner selects best per (B, IC, HW).
        triton.Config({'BLOCK_N': 32,  'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
    ],
    key=['B', 'IC', 'HW'],
)
@triton.jit
def _conv1x1_fused_kernel(
    input_ptr,   # [B, IC, HW]  (row-major, contiguous)
    weight_ptr,  # [OC, IC]     (row-major, contiguous)
    bias_ptr,    # [OC]
    output_ptr,  # [B, OC, HW]  (row-major, contiguous)
    B, OC, IC, HW,
    BLOCK_M: tl.constexpr,   # tile over OC – fixed to 32 (power-of-2 >= 17)
    BLOCK_N: tl.constexpr,   # tile over HW – autotuned
    BLOCK_K: tl.constexpr,   # tile over IC – autotuned
):
    # 2-D grid:  dim0 (x) → B*HW tiles (x-axis limit 2^31-1, safe for large B)
    #            dim1 (y) → OC tiles  (always 1, well within y-limit of 65535)
    pid_n = tl.program_id(0)   # B*HW tile index
    pid_m = tl.program_id(1)   # OC tile index

    # Identify which batch item and starting hw offset this CTA handles.
    # Because all BLOCK_N values (32,64,128,256) divide HW=4096 evenly,
    # every CTA stays within exactly one batch item.
    bhw_start = pid_n * BLOCK_N
    b_idx     = bhw_start // HW    # batch index (scalar)
    hw_start  = bhw_start  % HW    # starting hw position (scalar)

    oc_start = pid_m * BLOCK_M
    oc_offs  = oc_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    hw_offs  = hw_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    oc_mask = oc_offs < OC    # [BLOCK_M]
    hw_mask = hw_offs < HW    # [BLOCK_N]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over IC in BLOCK_K steps
    for k_start in range(0, IC, BLOCK_K):
        ic_offs = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]
        ic_mask = ic_offs < IC

        # Weight tile: [BLOCK_M, BLOCK_K]  –  weight[oc, ic]
        w = tl.load(
            weight_ptr + oc_offs[:, None] * IC + ic_offs[None, :],
            mask=oc_mask[:, None] & ic_mask[None, :],
            other=0.0,
        )

        # Input tile: [BLOCK_K, BLOCK_N]  –  input[b_idx, ic, hw]
        x = tl.load(
            input_ptr + b_idx * IC * HW + ic_offs[:, None] * HW + hw_offs[None, :],
            mask=ic_mask[:, None] & hw_mask[None, :],
            other=0.0,
        )

        # Fused multiply-accumulate: acc += w @ x
        acc += tl.dot(w, x)

    # Add bias  [OC]
    bias = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    # Store result: output[b_idx, oc, hw]
    tl.store(
        output_ptr + b_idx * OC * HW + oc_offs[:, None] * HW + hw_offs[None, :],
        acc.to(output_ptr.dtype.element_ty),
        mask=oc_mask[:, None] & hw_mask[None, :],
    )


# OC=17 → smallest power-of-2 that fits is 32
_BLOCK_M = 32


# ---------------------------------------------------------------------------
# Wrapper – must be decorated with @torch.fx.wrap
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _conv1x1_mul_reshape(in_0, in_1, in_2):
    """
    Fused replacement for:
        conv2d = torch.conv2d(in_2, in_1, in_0, (1,1), (0,0), (1,1), 1)
        out    = (conv2d * 1.0).reshape(-1, 17, 4096)
    2-D grid (B*HW tiles × OC tiles) with software-pipelined Triton kernel.
    """
    B  = in_2.shape[0]
    IC = in_2.shape[1]
    H  = in_2.shape[2]
    W  = in_2.shape[3]
    OC = in_1.shape[0]
    HW = H * W

    inp  = in_2.reshape(B, IC, HW)
    wgt  = in_1.reshape(OC, IC)
    bias = in_0

    if not inp.is_contiguous():
        inp = inp.contiguous()
    if not wgt.is_contiguous():
        wgt = wgt.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()

    out = torch.empty((B, OC, HW), dtype=in_2.dtype, device=in_2.device)

    # 2-D grid: dim0 (x) = B*HW tiles, dim1 (y) = OC tiles
    # x-axis limit is 2^31-1 (safe for large B); y=1 always (OC=17 ≤ BLOCK_M=32)
    grid = lambda meta: (
        triton.cdiv(B * HW, meta['BLOCK_N']),
        triton.cdiv(OC, _BLOCK_M),
    )

    _conv1x1_fused_kernel[grid](
        inp, wgt, bias, out,
        B, OC, IC, HW,
        BLOCK_M=_BLOCK_M,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func – zero-argument, returns the callable
# ---------------------------------------------------------------------------

def replacement_func():
    return _conv1x1_mul_reshape