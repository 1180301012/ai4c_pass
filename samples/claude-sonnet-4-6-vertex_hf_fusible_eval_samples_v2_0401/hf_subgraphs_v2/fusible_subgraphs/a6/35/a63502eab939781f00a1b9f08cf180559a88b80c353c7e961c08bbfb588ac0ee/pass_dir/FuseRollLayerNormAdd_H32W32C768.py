"""
Fused pass: contiguous -> view(-1,32,32,768) -> roll((4,4),(1,2)) -> view(1,1024,768)
            -> layer_norm((768,),weight,bias,1e-5) -> residual add

Works for both float16 and bfloat16 inputs.

The roll operation maps output row (h,w) -> source row ((h-4)%32, (w-4)%32).
We fuse this index remapping with layer_norm + residual add into one kernel.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 32, 32, 768)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 1024, 768)
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (768,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _fused_roll_ln_add_h32w32c768(
    in3_ptr,    # [1024, 768]  — in_3 after contiguous+reshape
    in2_ptr,    # [1024, 768]  — residual in_2
    w_ptr,      # [768]        — layer_norm weight (in_1)
    b_ptr,      # [768]        — layer_norm bias   (in_0)
    out_ptr,    # [1024, 768]  — output
    IS_BF16: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Shape constants (compile-time)
    H      = 32
    W      = 32
    C      = 768
    SHIFT  = 4
    EPS    = 1e-5

    row_id = tl.program_id(0)          # 0 … 1023

    # Compute source row after inverse-roll
    h     = row_id // W
    w     = row_id  % W
    src_h = (h - SHIFT + H) % H
    src_w = (w - SHIFT + W) % W
    src_row = src_h * W + src_w

    offsets = tl.arange(0, BLOCK_C)    # 0 … 1023
    mask    = offsets < C              # True for first 768

    # ---- Load in3 from source row ----------------------------------------
    x = tl.load(in3_ptr + src_row * C + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # ---- Layer Norm ----------------------------------------------------------
    # One-pass mean + variance using PyTorch's formula:
    #   var = E[x^2] - E[x]^2  (matches PyTorch CUDA kernel exactly)
    # Use hardware rsqrt to match PyTorch's rsqrt() call.
    sum_val = tl.sum(x_f32, axis=0)
    sum_sq  = tl.sum(x_f32 * x_f32, axis=0)
    mean    = sum_val / C
    var     = sum_sq / C - mean * mean

    inv_std = tl.math.rsqrt(var + EPS)
    # Split into two ops to avoid FMA contraction (matches PyTorch's two-step pattern)
    x_centered = x_f32 - mean
    x_norm  = x_centered * inv_std

    # scale + shift
    weight = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias   = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_out  = x_norm * weight + bias

    # ---- Residual add -------------------------------------------------------
    # Cast layer_norm result to native dtype first (matching PyTorch's order:
    # layer_norm returns fp16/bf16, then add residual in fp16/bf16)
    if IS_BF16:
        x_out_native = x_out.to(tl.bfloat16)
    else:
        x_out_native = x_out.to(tl.float16)

    # Load residual in native dtype (no fp32 upcast)
    residual = tl.load(in2_ptr + row_id * C + offsets, mask=mask, other=0.0)

    # Add in native dtype and store
    result = x_out_native + residual
    tl.store(out_ptr + row_id * C + offsets, result, mask=mask)


# Lightweight roll-only kernel: exact indexed copy used inside the wrapper
@triton.jit
def _roll_only_h32w32c768(src_ptr, dst_ptr, BLOCK_C: tl.constexpr):
    H = 32; W = 32; C = 768; SHIFT = 4
    row_id = tl.program_id(0)
    h = row_id // W;  w = row_id % W
    src_h = (h - SHIFT + H) % H
    src_w = (w - SHIFT + W) % W
    src_row = src_h * W + src_w
    offsets = tl.arange(0, BLOCK_C)
    mask = offsets < C
    data = tl.load(src_ptr + src_row * C + offsets, mask=mask)
    tl.store(dst_ptr + row_id * C + offsets, data, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_roll_ln_add_h32w32c768(in_0, in_1, in_2, in_3):
    H, W, C = 32, 32, 768
    N_rows  = H * W          # 1024

    in3_flat = in_3.contiguous().reshape(N_rows, C)
    in2_flat = in_2.reshape(N_rows, C)
    out      = torch.empty_like(in_2)
    out_flat = out.reshape(N_rows, C)
    is_bf16  = (in_2.dtype == torch.bfloat16)

    _fused_roll_ln_add_h32w32c768[(N_rows,)](
        in3_ptr=in3_flat, in2_ptr=in2_flat,
        w_ptr=in_1, b_ptr=in_0, out_ptr=out_flat,
        IS_BF16=is_bf16, BLOCK_C=1024,
        num_warps=16, num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# Replacement entry-point
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_roll_ln_add_h32w32c768