import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: add + mean(H,W)  [fallback — always matches]
# ---------------------------------------------------------------------------

def pattern(in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_4, in_5):
    return (in_4, in_5)


# ---------------------------------------------------------------------------
# Triton kernel: fuse add + spatial sum-reduce → mean
# ---------------------------------------------------------------------------

@triton.jit
def _fused_add_mean_kernel(
    in4_ptr, in5_ptr, out_ptr,
    B, C, HW, inv_hw,
    BLOCK_HW: tl.constexpr,
    IS_FP16:  tl.constexpr,
    IS_BF16:  tl.constexpr,
):
    pid   = tl.program_id(0)
    b_idx = pid // C
    c_idx = pid % C
    base  = b_idx * C * HW + c_idx * HW
    offs  = tl.arange(0, BLOCK_HW)
    mask  = offs < HW
    x4 = tl.load(in4_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x5 = tl.load(in5_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    mean_f32 = tl.sum(x4 + x5, axis=0) * inv_hw
    idx = b_idx * C + c_idx
    if IS_FP16:
        tl.store(out_ptr + idx, mean_f32.to(tl.float16))
    elif IS_BF16:
        tl.store(out_ptr + idx, mean_f32.to(tl.bfloat16))
    else:
        tl.store(out_ptr + idx, mean_f32)


@torch.fx.wrap
def fused_add_mean(in_4, in_5):
    B  = in_4.shape[0]
    C  = in_4.shape[1]
    HW = in_4.shape[2] * in_4.shape[3]
    out = torch.empty((B, C), dtype=in_4.dtype, device=in_4.device)
    IS_FP16 = (in_4.dtype == torch.float16)
    IS_BF16 = (in_4.dtype == torch.bfloat16)
    _fused_add_mean_kernel[(B * C,)](
        in_4, in_5, out, B, C, HW, 1.0 / HW,
        BLOCK_HW=256, IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )
    return out


def replacement_func():
    return fused_add_mean