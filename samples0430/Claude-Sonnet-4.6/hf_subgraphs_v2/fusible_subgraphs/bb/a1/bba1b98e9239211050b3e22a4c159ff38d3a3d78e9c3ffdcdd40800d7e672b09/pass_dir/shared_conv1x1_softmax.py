"""
Shared Triton kernels for fused 1x1 conv + softmax.

Pattern:
  out = conv2d(x, w, b, stride=(1,1), pad=(0,0), dil=(1,1), groups=1)
       .view(B, 1, -1)
       .softmax(dim=-1)

Where w has shape [1, C, 1, 1] — single output channel, so the conv is a
channel-wise dot product at each spatial position. We fuse:
  (1) reduction over C=512 channels  (conv1x1_reduce_kernel)
  (2) softmax over HW=4096 positions (softmax_kernel)
into a two-kernel sequence that avoids materialising the [B,1,H,W] intermediate.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1 – 1×1 conv reduction
# Grid: (B * ceil(HW / BLOCK_HW),)
# Each program handles BLOCK_HW spatial positions for one batch element.
# Inner loop tiles over C channels in BLOCK_C steps.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 64},  num_warps=8),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_C': 64}, num_warps=8),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def _conv1x1_reduce_kernel(
    in2_ptr,   # [B, C, HW] contiguous (NCHW with HW = H*W)
    w_ptr,     # [C]         weight, flattened from [1,C,1,1]
    bias_ptr,  # [1]         bias scalar
    out_ptr,   # [B*HW]      float32 intermediate
    B, C, HW,
    BLOCK_HW: tl.constexpr,
    BLOCK_C:  tl.constexpr,
):
    pid       = tl.program_id(0)
    num_tiles = tl.cdiv(HW, BLOCK_HW)
    b         = pid // num_tiles
    tile      = pid %  num_tiles

    hw_start  = tile * BLOCK_HW
    hw_offs   = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask   = hw_offs < HW

    acc       = tl.zeros([BLOCK_HW], dtype=tl.float32)
    in2_base  = b.to(tl.int64) * C * HW

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C

        # Load weight tile [BLOCK_C]
        w = tl.load(w_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        # Load input tile [BLOCK_C, BLOCK_HW]
        # in2[b, c, hw] = in2_base + c * HW + hw
        ptrs_2d = in2_base + c_offs[:, None].to(tl.int64) * HW + hw_offs[None, :].to(tl.int64)
        x = tl.load(in2_ptr + ptrs_2d,
                    mask=c_mask[:, None] & hw_mask[None, :],
                    other=0.0).to(tl.float32)

        # Dot product: accumulate channel dimension
        acc += tl.sum(x * w[:, None], axis=0)

    # Add bias
    bias  = tl.load(bias_ptr).to(tl.float32)
    acc  += bias

    # Store [B, HW] intermediate
    out_offs = b.to(tl.int64) * HW + hw_offs.to(tl.int64)
    tl.store(out_ptr + out_offs, acc, mask=hw_mask)


# ---------------------------------------------------------------------------
# Kernel 2 – row-wise softmax over HW positions
# Grid: (B,)
# BLOCK_HW must equal HW (= 4096 for 64×64 spatial) so all elements fit in
# one program; no masking needed.
# ---------------------------------------------------------------------------

@triton.jit
def _softmax_kernel(
    inp_ptr,              # [B*HW] float32 intermediate
    out_ptr,              # [B*HW] output in DTYPE
    BLOCK_HW: tl.constexpr,
    DTYPE:    tl.constexpr,
):
    b        = tl.program_id(0)
    hw_offs  = tl.arange(0, BLOCK_HW)

    # Load float32 logits
    x        = tl.load(inp_ptr + b.to(tl.int64) * BLOCK_HW + hw_offs)

    # Numerically-stable softmax
    max_x    = tl.max(x, axis=0)
    exp_x    = tl.exp(x - max_x)
    sum_x    = tl.sum(exp_x, axis=0)
    result   = (exp_x / sum_x).to(DTYPE)

    tl.store(out_ptr + b.to(tl.int64) * BLOCK_HW + hw_offs, result)


# ---------------------------------------------------------------------------
# Host wrapper – returned by replacement_func()
# Arguments match replacement_args order: (bias=in_0, weight=in_1, x=in_2)
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    torch.float32:  tl.float32,
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@torch.fx.wrap
def fused_conv1x1_softmax(bias, weight, x):
    """
    bias   : [1]                 (in_0)
    weight : [1, C, 1, 1]       (in_1)
    x      : [B, C, H, W]       (in_2)
    returns: [B, 1, H*W]        softmax output in same dtype as x
    """
    B   = x.shape[0]
    C   = x.shape[1]
    H   = x.shape[2]
    W   = x.shape[3]
    HW  = H * W          # always 4096 for these graphs

    dtype = x.dtype
    DTYPE = _DTYPE_MAP[dtype]

    # Float32 intermediate buffer for numerically stable softmax
    tmp = torch.empty(B * HW, dtype=torch.float32, device=x.device)

    # --- Kernel 1: 1×1 conv reduction ---
    grid_conv = lambda META: (B * triton.cdiv(HW, META['BLOCK_HW']),)
    _conv1x1_reduce_kernel[grid_conv](
        x, weight, bias, tmp,
        B, C, HW,
    )

    # --- Kernel 2: softmax ---
    out = torch.empty(B * HW, dtype=dtype, device=x.device)
    _softmax_kernel[(B,)](
        tmp, out,
        BLOCK_HW=HW,
        DTYPE=DTYPE,
    )

    return out.view(B, 1, HW)