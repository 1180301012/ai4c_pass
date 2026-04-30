"""
Shared Triton kernels for channel shuffle operations.

Channel shuffle for [B, C, H, W] source pair (A, B):
  - Concatenate: src = cat([A, B], dim=1) -> [B, 2C, H, W]
  - view(B, 2, C, H, W) -> transpose(1,2) -> contiguous() -> view(B, 2C, H, W)
  - chunk(2, dim=1) -> (out0[B,C,H,W], out1[B,C,H,W])

Result:
  out0[b, j, h, w] = A[b, j//2, h, w] if j%2==0, B[b, j//2, h, w] if j%2==1  (j=0..C-1)
  out1[b, j, h, w] = A[b, C//2+j//2, h, w] if j%2==0, B[b, C//2+j//2, h, w] if j%2==1
"""

import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Stream 1 kernel  (in_2, in_4  →  out1a, out1b)
# Shapes: A,B in [B, 20, 64, 48],  out1a, out1b in [B, 20, 64, 48]
# HALF_C=10, HW=3072
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['HW'],
)
@triton.jit
def _channel_shuffle_s1(
    in2_ptr, in4_ptr,
    out1a_ptr, out1b_ptr,
    B, C_in, HW,
    HALF_C,
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)
    pid_b = pid // (C_in * 2)
    pid_c = (pid % (C_in * 2))

    c_out   = pid_c // 2           # output channel index 0..C_in-1
    c_in    = pid_c % 2            # source pair: 0 = in_2, 1 = in_4
    c_src   = c_out                 # source channel 0..C_in-1

    hw_off  = pid_b * (C_in * HW) + c_src * HW + c_src * HW  # base for (b, c_src)
    hw_off2 = pid_b * (C_in * HW) + (c_src + HALF_C) * HW + (c_src + HALF_C) * HW

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < HW

    val_a = tl.load(in2_ptr + hw_off   + offsets, mask=mask, other=0.0)
    val_b = tl.load(in4_ptr + hw_off   + offsets, mask=mask, other=0.0)
    val_a2 = tl.load(in2_ptr + hw_off2 + offsets, mask=mask, other=0.0)
    val_b2 = tl.load(in4_ptr + hw_off2 + offsets, mask=mask, other=0.0)

    tl.store(out1a_ptr + pid * HW + offsets, tl.where(c_in == 0, val_a,  val_b),  mask=mask)
    tl.store(out1b_ptr + pid * HW + offsets, tl.where(c_in == 0, val_a2, val_b2), mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Stream 2 kernel  (in_3, tmp_4  →  out2a, out2b)
# Shapes: A,B in [B, 40, 32, 24],  out2a, out2b in [B, 40, 32, 24]
# HALF_C=20, HW=768
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['HW'],
)
@triton.jit
def _channel_shuffle_s2(
    in3_ptr, tmp4_ptr,
    out2a_ptr, out2b_ptr,
    B, C_in, HW,
    HALF_C,
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)
    pid_b = pid // (C_in * 2)
    pid_c = (pid % (C_in * 2))

    c_out   = pid_c // 2
    c_in    = pid_c % 2
    c_src   = c_out

    hw_off  = pid_b * (C_in * HW) + c_src * HW + c_src * HW
    hw_off2 = pid_b * (C_in * HW) + (c_src + HALF_C) * HW + (c_src + HALF_C) * HW

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < HW

    val_a  = tl.load(in3_ptr  + hw_off   + offsets, mask=mask, other=0.0)
    val_b  = tl.load(tmp4_ptr + hw_off   + offsets, mask=mask, other=0.0)
    val_a2 = tl.load(in3_ptr  + hw_off2 + offsets, mask=mask, other=0.0)
    val_b2 = tl.load(tmp4_ptr + hw_off2 + offsets, mask=mask, other=0.0)

    tl.store(out2a_ptr + pid * HW + offsets, tl.where(c_in == 0, val_a,  val_b),  mask=mask)
    tl.store(out2b_ptr + pid * HW + offsets, tl.where(c_in == 0, val_a2, val_b2), mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: sigmoid(x)
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _sigmoid_fp16_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    x       = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # upcast to fp32 for numerical stability, then cast back
    x_f32   = x.to(tl.float32)
    sig_f32 = 1.0 / (1.0 + tl.exp(-x_f32))
    tl.store(out_ptr + offsets, sig_f32.to(x.dtype), mask=mask)


@triton.jit
def _sigmoid_fp32_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    x       = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32   = x.to(tl.float32)
    sig_f32 = 1.0 / (1.0 + tl.exp(-x_f32))
    tl.store(out_ptr + offsets, sig_f32, mask=mask)


@triton.jit
def _sigmoid_bf16_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    x       = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32   = x.to(tl.float32)
    sig_f32 = 1.0 / (1.0 + tl.exp(-x_f32))
    tl.store(out_ptr + offsets, sig_f32.to(x.dtype), mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Public Python wrapper  (called from the @torch.fx.wrap wrapper in each pass)
# ──────────────────────────────────────────────────────────────────────────────

def run_channel_shuffle_stream1(A, B, out1a, out1b, B_batch):
    """A, B: [B, 20, 64, 48]  →  out1a, out1b: [B, 20, 64, 48]"""
    C_in   = 20
    HW     = 3072
    HALF_C = 10
    n_programs = B_batch * C_in * 2
    grid = lambda meta: (triton.cdiv(HW, meta['BLOCK_SIZE']),)
    _channel_shuffle_s1[grid](
        A, B, out1a, out1b,
        B_batch, C_in, HW, HALF_C,
        # BLOCK_SIZE is injected by @triton.autotune — do NOT pass it here
    )


def run_channel_shuffle_stream2(A, B, out2a, out2b, B_batch):
    """A, B: [B, 40, 32, 24]  →  out2a, out2b: [B, 40, 32, 24]"""
    C_in   = 40
    HW     = 768
    HALF_C = 20
    grid = lambda meta: (triton.cdiv(HW, meta['BLOCK_SIZE']),)
    _channel_shuffle_s2[grid](
        A, B, out2a, out2b,
        B_batch, C_in, HW, HALF_C,
        # BLOCK_SIZE is injected by @triton.autotune — do NOT pass it here
    )


def run_sigmoid(x, out):
    """elementwise sigmoid in Triton; handles fp16, bf16, fp32."""
    n     = x.numel()
    BLOCK = 1024
    grid  = (triton.cdiv(n, BLOCK),)
    if x.dtype == torch.float16:
        _sigmoid_fp16_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    elif x.dtype == torch.float32:
        _sigmoid_fp32_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    else:  # bfloat16
        _sigmoid_bf16_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)