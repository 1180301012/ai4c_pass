"""
Shared Triton kernels and dispatch wrapper for all passes.
All pass files import this module so replacement_func is identical.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: fuse add + spatial mean  [B,C,H,W] -> [B,C]
# ---------------------------------------------------------------------------
@triton.jit
def _add_mean_k(
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


# ---------------------------------------------------------------------------
# Kernel B: fuse add + spatial mean + BN inference
#           [B,C,H,W] -> mean [B,C]  +  BN_out [B,C]
# ---------------------------------------------------------------------------
@triton.jit
def _add_mean_bn_k(
    in4_ptr, in5_ptr,
    rm_ptr, rv_ptr, w_ptr, b_ptr,
    mean_ptr, bn_ptr,
    B, C, HW, inv_hw, eps,
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
    rm    = tl.load(rm_ptr + c_idx).to(tl.float32)
    rv    = tl.load(rv_ptr + c_idx).to(tl.float32)
    w     = tl.load(w_ptr  + c_idx).to(tl.float32)
    b_val = tl.load(b_ptr  + c_idx).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(rv + eps)
    bn_f32  = (mean_f32 - rm) * inv_std * w + b_val
    idx = b_idx * C + c_idx
    if IS_FP16:
        tl.store(mean_ptr + idx, mean_f32.to(tl.float16))
        tl.store(bn_ptr   + idx, bn_f32.to(tl.float16))
    elif IS_BF16:
        tl.store(mean_ptr + idx, mean_f32.to(tl.bfloat16))
        tl.store(bn_ptr   + idx, bn_f32.to(tl.bfloat16))
    else:
        tl.store(mean_ptr + idx, mean_f32)
        tl.store(bn_ptr   + idx, bn_f32)


# ---------------------------------------------------------------------------
# Python wrappers (not @torch.fx.wrap — called from dispatch_replacement)
# ---------------------------------------------------------------------------

def _run_add_mean(a4, a5):
    B  = a4.shape[0]
    C  = a4.shape[1]
    HW = a4.shape[2] * a4.shape[3]
    BHW = triton.next_power_of_2(HW)
    out = torch.empty((B, C), dtype=a4.dtype, device=a4.device)
    IS_FP16 = (a4.dtype == torch.float16)
    IS_BF16 = (a4.dtype == torch.bfloat16)
    _add_mean_k[(B * C,)](
        a4, a5, out, B, C, HW, 1.0 / HW,
        BLOCK_HW=BHW, IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )
    return out


def _run_add_mean_bn(a0, a1, a2, a3, a4, a5):
    # a0=rm, a1=rv, a2=bias, a3=weight, a4=in4, a5=in5
    B  = a4.shape[0]
    C  = a4.shape[1]
    HW = a4.shape[2] * a4.shape[3]
    BHW = triton.next_power_of_2(HW)
    mean_out = torch.empty((B, C), dtype=a4.dtype, device=a4.device)
    bn_out   = torch.empty((B, C), dtype=a4.dtype, device=a4.device)
    IS_FP16 = (a4.dtype == torch.float16)
    IS_BF16 = (a4.dtype == torch.bfloat16)
    _add_mean_bn_k[(B * C,)](
        a4, a5,
        a0, a1, a3, a2,   # rm, rv, weight, bias
        mean_out, bn_out,
        B, C, HW, 1.0 / HW, 1e-05,
        BLOCK_HW=BHW, IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )
    return (bn_out, mean_out)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper (IDENTICAL across all pass files — fixes
# output_pass_replacement_func_limit deduplication).
# Signature: 6 tensor slots + 1 route string.
# For the add-mean-only pass, a0..a3 are set to duplicates of a4/a5.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_replacement(a0, a1, a2, a3, a4, a5, route):
    if route == "route_full_bn":
        return _run_add_mean_bn(a0, a1, a2, a3, a4, a5)
    elif route == "route_legit_bn":
        return _run_add_mean_bn(a0, a1, a2, a3, a4, a5)
    else:                           # route_add_mean (fallback)
        return _run_add_mean(a4, a5)