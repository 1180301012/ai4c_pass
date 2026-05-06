import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['HW'],
)
@triton.jit
def fused_softmax_scale_sum_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """
    Each program handles (b=0, c, hw_block).
    Grid: (C, ceil(HW / BLOCK_SIZE))

    Computes:
      scale0 = softmax(in_1[0,0,...,h,w])[0],
      scale1 = softmax(in_1[0,1,...,h,w])[0],
      out[0,0,c,h,w] = in_0[0,0,c,h,w]*scale0 + in_0[0,1,c,h,w]*scale1
    """
    pid_b = tl.program_id(0)   # channel c in [0, C)
    pid_hw = tl.program_id(1)  # spatial block

    # ---- Load in_1[b=0, c=pid_b, 0:2, h, w] ----
    # in_1 has shape [1, 2, C, 1, 1] and is contiguous.
    # Flat offset for [0, 0, c, 0, 0] = c, for [0, 1, c, 0, 0] = C + c.
    in1_base = pid_b * 2
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    in1_0 = tl.load(in1_ptr + in1_base + offsets, mask=mask, other=float('-inf'))
    in1_1 = tl.load(in1_ptr + in1_base + HW + offsets, mask=mask, other=float('-inf'))

    # Compute softmax over dim=1 (the 2 values)
    in1_0 = in1_0.to(tl.float32)
    in1_1 = in1_1.to(tl.float32)
    in1_max = tl.maximum(in1_0, in1_1)
    in1_0 = tl.exp(in1_0 - in1_max)
    in1_1 = tl.exp(in1_1 - in1_max)
    scale0 = in1_0 / (in1_0 + in1_1)
    scale1 = in1_1 / (in1_0 + in1_1)

    scale0 = scale0.to(DTYPE)
    scale1 = scale1.to(DTYPE)

    # ---- Load in_0[0, 0, c, hw_block] and in_0[0, 1, c, hw_block] ----
    # in_0 has shape [1, 2, C, H, W] and is contiguous.
    # Strides: [2*C*HW, C*HW, HW, W, 1]
    # in_0[0, 0, c, hw] flat = c*HW + hw
    # in_0[0, 1, c, hw] flat = C*HW + c*HW + hw
    in0_base = pid_b * HW
    in0_0 = tl.load(in0_ptr + in0_base + pid_hw * BLOCK_SIZE + offsets, mask=mask, other=0.0)
    in0_1 = tl.load(in0_ptr + in0_base + HW + pid_hw * BLOCK_SIZE + offsets, mask=mask, other=0.0)

    # ---- Compute weighted sum and store ----
    result = in0_0 * scale0 + in0_1 * scale1

    # Output: shape [1, 1, C, H, W], strides [C*HW, C*HW, HW, W, 1]
    # out[0, 0, c, hw] flat = c*HW + hw  (same as in_0[0, 0, c, hw])
    tl.store(out_ptr + in0_base + pid_hw * BLOCK_SIZE + offsets, result, mask=mask)


@torch.fx.wrap
def fused_softmax_scale_sum(in_0, in_1):
    """
    Fused replacement for:
      tmp_0 = torch.softmax(in_1, dim=1)   # [1,2,C,1,1] -> [1,1,C,1,1]
      tmp_1 = in_0 * tmp_0                  # [1,2,C,H,W]
      tmp_2 = torch.sum(tmp_1, dim=1)       # [1,1,C,H,W]
    Returns (tmp_2,) as the model does.
    """
    # in_0: [1, 2, C, H, W]
    # in_1: [1, 2, C, 1, 1]
    batch, C, H, W = in_0.shape[0], in_0.shape[2], in_0.shape[3], in_0.shape[4]
    HW = H * W

    out = torch.empty(batch, 1, C, H, W, dtype=in_0.dtype, device=in_0.device)

    CFlat = batch * C

    DTYPE_BF16 = tl.bfloat16
    DTYPE_F16  = tl.float16
    DTYPE_F32  = tl.float32

    if in_0.dtype == torch.bfloat16:
        DTYPE = DTYPE_BF16
    elif in_0.dtype == torch.float16:
        DTYPE = DTYPE_F16
    else:
        DTYPE = DTYPE_F32

    grid = (CFlat, triton.cdiv(HW, 256))

    fused_softmax_scale_sum_kernel[grid](
        in_0, in_1, out,
        HW,
        DTYPE=DTYPE,
    )

    return (out,)


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_softmax_scale_sum