import torch
import triton
import triton.language as tl
import math


def pattern(conv1d_out, avg_pool_out, ln_weight, ln_bias):
    tmp_4 = torch.nn.functional.gelu(conv1d_out)
    tmp_7 = tmp_4[(Ellipsis, slice(None, 124, None))]
    tmp_6 = avg_pool_out[(Ellipsis, slice(None, 124, None))]
    tmp_8 = tmp_6 + tmp_7
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), ln_weight, ln_bias, 1e-05)
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
    return (tmp_11,)


def replacement_args(conv1d_out, avg_pool_out, ln_weight, ln_bias):
    return (conv1d_out, avg_pool_out, ln_weight, ln_bias)


# ============================================================
# Triton helper: tanh using exp (since tl.math.tanh may not exist)
# ============================================================
# tanh(x) = 2 / (1 + exp(-2x)) - 1

@triton.jit
def tanh_approx(x):
    # Clamp to avoid overflow: for |x| > 10, tanh ≈ ±1
    x_safe = tl.where(x > 10.0, 10.0, tl.where(x < -10.0, -10.0, x))
    exp_neg_2x = tl.math.exp(-2.0 * x_safe)
    return 2.0 / (1.0 + exp_neg_2x) - 1.0


# ============================================================
# Triton Kernel 1: GELU + Slice + Add + Transpose
# ============================================================
# Input: conv1d_out [1, 768, 125] (we use [:124])
# Input: avg_pool_out [1, 768, 124_or_125] (we use [:124])
# Output: intermediate [1, 124, 768] in transposed layout
# ============================================================

@triton.jit
def fused_gelu_slice_add_transpose_kernel(
    conv1d_ptr,        # pointer to conv1d output [1, N_CHANNELS, CONV_SEQ_LEN]
    avg_pool_ptr,      # pointer to avg_pool output [1, N_CHANNELS, POOL_SEQ_LEN]
    intermediate_ptr,  # pointer to intermediate [1, OUT_SEQ_LEN, N_CHANNELS] (transposed layout)
    CONV_SEQ_LEN: tl.constexpr,   # 125
    POOL_SEQ_LEN: tl.constexpr,   # 124 or 125
    OUT_SEQ_LEN: tl.constexpr,    # 124
    N_CHANNELS: tl.constexpr,     # 768
    BLOCK_C: tl.constexpr,        # channel tile size
    STORE_DTYPE: tl.constexpr,    # output dtype
):
    t = tl.program_id(0)   # output sequence position (0..OUT_SEQ_LEN-1)
    cb = tl.program_id(1)  # channel block index
    
    c_start = cb * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < N_CHANNELS
    
    # Load conv1d output at position t for this block of channels
    # conv1d_out[0, c, t] = conv1d_ptr + c * CONV_SEQ_LEN + t
    # Since conv1d_out is [1, N_CHANNELS, CONV_SEQ_LEN], we slice by only accessing t < OUT_SEQ_LEN
    conv_val = tl.load(conv1d_ptr + c_offsets * CONV_SEQ_LEN + t,
                       mask=c_mask,
                       other=0.0).to(tl.float32)
    
    # Apply GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    inner = sqrt_2_over_pi * (conv_val + 0.044715 * conv_val * conv_val * conv_val)
    gelu_val = 0.5 * conv_val * (1.0 + tanh_approx(inner))
    
    # Load avg_pool output at position t for this block of channels
    # avg_pool_out[0, c, t] = avg_pool_ptr + c * POOL_SEQ_LEN + t
    avg_val = tl.load(avg_pool_ptr + c_offsets * POOL_SEQ_LEN + t,
                      mask=c_mask,
                      other=0.0).to(tl.float32)
    
    # Add: gelu + avg_pool
    add_val = gelu_val + avg_val
    
    # Store to intermediate in transposed layout: intermediate[0, t, c]
    # intermediate[0, t, c] = intermediate_ptr + t * N_CHANNELS + c
    tl.store(intermediate_ptr + t * N_CHANNELS + c_offsets, add_val.to(STORE_DTYPE), mask=c_mask)


# ============================================================
# Triton Kernel 2: Layer Normalization
# ============================================================
# Input: intermediate [1, 124, 768]
# Input: ln_weight [768], ln_bias [768]
# Output: output [1, 124, 768]
# ============================================================

@triton.jit
def layer_norm_kernel(
    intermediate_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    output_ptr,
    OUT_SEQ_LEN: tl.constexpr,
    N_CHANNELS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STORE_DTYPE: tl.constexpr,
):
    t = tl.program_id(0)  # sequence position (0..OUT_SEQ_LEN-1)
    
    # First pass: compute mean and variance
    acc_sum = 0.0
    acc_sum_sq = 0.0
    
    for c_start in range(0, N_CHANNELS, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < N_CHANNELS
        
        val = tl.load(intermediate_ptr + t * N_CHANNELS + c_offsets,
                       mask=c_mask, other=0.0).to(tl.float32)
        acc_sum += tl.sum(val * c_mask.to(tl.float32), axis=0)
        acc_sum_sq += tl.sum(val * val * c_mask.to(tl.float32), axis=0)
    
    mean = acc_sum / N_CHANNELS
    var = acc_sum_sq / N_CHANNELS - mean * mean
    
    # Second pass: normalize and write output
    for c_start in range(0, N_CHANNELS, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < N_CHANNELS
        
        val = tl.load(intermediate_ptr + t * N_CHANNELS + c_offsets,
                       mask=c_mask, other=0.0).to(tl.float32)
        ln_w = tl.load(ln_weight_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        ln_b = tl.load(ln_bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        
        normalized = (val - mean) / tl.sqrt(var + EPS)
        output_val = normalized * ln_w + ln_b
        
        tl.store(output_ptr + t * N_CHANNELS + c_offsets,
                 output_val.to(STORE_DTYPE), mask=c_mask)


# ============================================================
# Kernel Wrapper
# ============================================================

@torch.fx.wrap
def fused_gelu_add_layernorm_wrapper(conv1d_out, avg_pool_out, ln_weight, ln_bias):
    """
    Fused computation: GELU(conv1d_out)[:124] + avg_pool_out[:124], transpose, layer_norm
    Dropout with training=False is identity, so skipped.
    """
    # Determine dimensions
    N_CHANNELS = 768
    OUT_SEQ_LEN = 124
    CONV_SEQ_LEN = conv1d_out.shape[2]  # 125
    POOL_SEQ_LEN = avg_pool_out.shape[2]  # 124 (or could be 125)
    
    # Determine output dtype
    input_dtype = conv1d_out.dtype
    if input_dtype == torch.float16:
        STORE_DTYPE = tl.float16
    elif input_dtype == torch.bfloat16:
        STORE_DTYPE = tl.bfloat16
    else:
        STORE_DTYPE = tl.float32
    
    # Allocate intermediate buffer [1, OUT_SEQ_LEN, N_CHANNELS]
    intermediate = torch.empty((1, OUT_SEQ_LEN, N_CHANNELS), dtype=input_dtype, device=conv1d_out.device)
    
    # Allocate output [1, OUT_SEQ_LEN, N_CHANNELS]
    output = torch.empty((1, OUT_SEQ_LEN, N_CHANNELS), dtype=input_dtype, device=conv1d_out.device)
    
    # Kernel 1: GELU + slice + add + transpose
    BLOCK_C_1 = 64  # 768 / 64 = 12 blocks
    grid1 = (OUT_SEQ_LEN, N_CHANNELS // BLOCK_C_1)
    fused_gelu_slice_add_transpose_kernel[grid1](
        conv1d_ptr=conv1d_out,
        avg_pool_ptr=avg_pool_out,
        intermediate_ptr=intermediate,
        CONV_SEQ_LEN=CONV_SEQ_LEN,
        POOL_SEQ_LEN=POOL_SEQ_LEN,
        OUT_SEQ_LEN=OUT_SEQ_LEN,
        N_CHANNELS=N_CHANNELS,
        BLOCK_C=BLOCK_C_1,
        STORE_DTYPE=STORE_DTYPE,
    )
    
    # Kernel 2: layer_norm
    BLOCK_C_2 = 128  # 768 / 128 = 6 iterations
    grid2 = (OUT_SEQ_LEN,)
    layer_norm_kernel[grid2](
        intermediate_ptr=intermediate,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        output_ptr=output,
        OUT_SEQ_LEN=OUT_SEQ_LEN,
        N_CHANNELS=N_CHANNELS,
        EPS=1e-5,
        BLOCK_C=BLOCK_C_2,
        STORE_DTYPE=STORE_DTYPE,
    )
    
    return (output,)


def replacement_func():
    return fused_gelu_add_layernorm_wrapper