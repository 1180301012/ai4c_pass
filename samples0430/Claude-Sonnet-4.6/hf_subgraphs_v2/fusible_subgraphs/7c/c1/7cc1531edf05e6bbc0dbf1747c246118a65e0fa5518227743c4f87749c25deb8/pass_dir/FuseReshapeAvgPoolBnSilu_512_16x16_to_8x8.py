"""
Fused pass: reshape + avg_pool2d(2x2,s=2) + batch_norm(inference) + silu
Covers both bfloat16 and float16 variants for input shape [4,128,256] -> [1,512,8,8].
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ---------------------------------------------------------------------------
# Fused Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_avgpool_bn_silu_kernel(
    input_ptr,    # [1, 512, 16, 16]  (contiguous; same layout as [4,128,256])
    mean_ptr,     # [512] float32 on GPU
    var_ptr,      # [512] float32 on GPU
    weight_ptr,   # [512] float32 on GPU  (BN weight)
    bias_ptr,     # [512] float32 on GPU  (BN bias)
    output_ptr,   # [1, 512, 8, 8]
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode flat output index -> [c, oh, ow]
    # n_elements = 512 * 8 * 8 = 32 768
    # H_out=8, W_out=8  -> spatial stride = 64
    c   = offsets // 64           # channel   [0, 511]
    hw  = offsets % 64
    oh  = hw // 8                 # output row [0, 7]
    ow  = hw % 8                  # output col [0, 7]

    # Per-channel BN params (float32)
    mean   = tl.load(mean_ptr   + c, mask=mask, other=0.0)
    var    = tl.load(var_ptr    + c, mask=mask, other=1.0)
    w      = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b      = tl.load(bias_ptr   + c, mask=mask, other=0.0)

    # 2x2 avg-pool from the [1, 512, 16, 16] input
    # Strides: channel stride = 16*16 = 256, row stride = 16, col stride = 1
    ih   = oh * 2
    iw   = ow * 2
    base = c * 256                # channel base offset in flat input

    x00 = tl.load(input_ptr + base + ih * 16 + iw,           mask=mask, other=0.0).to(tl.float32)
    x01 = tl.load(input_ptr + base + ih * 16 + iw + 1,       mask=mask, other=0.0).to(tl.float32)
    x10 = tl.load(input_ptr + base + (ih + 1) * 16 + iw,     mask=mask, other=0.0).to(tl.float32)
    x11 = tl.load(input_ptr + base + (ih + 1) * 16 + iw + 1, mask=mask, other=0.0).to(tl.float32)

    avg = (x00 + x01 + x10 + x11) * 0.25

    # Batch norm (inference): y = (x - mean) * rsqrt(var + eps) * weight + bias
    bn_out = (avg - mean) * tl.rsqrt(var + eps) * w + b

    # SiLU: x * sigmoid(x)
    silu_out = bn_out * tl.sigmoid(bn_out)

    # Store – Triton auto-converts float32 to the output tensor's dtype (bf16/fp16)
    tl.store(output_ptr + offsets, silu_out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_avgpool_bn_silu(in_0, in_1, in_2, in_3, in_4):
    """
    Drop-in replacement for:
        tmp_4 = in_4.reshape(1, 512, 16, 16)
        tmp_5 = avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
        tmp_6 = batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
        tmp_7 = silu(tmp_6, inplace=True)
    """
    device = in_4.device

    # BN parameters live on CPU; bring them to the compute device as float32
    mean   = torch.as_tensor(in_0, device=device, dtype=torch.float32)
    var    = torch.as_tensor(in_1, device=device, dtype=torch.float32)
    weight = torch.as_tensor(in_3, device=device, dtype=torch.float32)
    bias   = torch.as_tensor(in_2, device=device, dtype=torch.float32)

    # Output tensor: [1, 512, 8, 8], same dtype as in_4
    output = torch.empty(1, 512, 8, 8, device=device, dtype=in_4.dtype)

    n_elements = 512 * 8 * 8   # 32 768

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_avgpool_bn_silu_kernel[grid](
        in_4,                    # [4,128,256] contiguous == [1,512,16,16] in memory
        mean, var, weight, bias,
        output,
        n_elements,
        eps=1e-5,
    )

    return output


# ---------------------------------------------------------------------------
# replacement_func: zero-argument, returns the callable
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_avgpool_bn_silu