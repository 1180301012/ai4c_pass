import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_gelu_pool_add_ln_kernel(
    in3_ptr, weight_ptr, conv_bias_ptr, ln_weight_ptr, ln_bias_ptr, out_ptr,
    C:       tl.constexpr,   # 768  channels
    K:       tl.constexpr,   # 48   kernel size
    BLOCK_C: tl.constexpr,   # 256
):
    pid_r  = tl.program_id(0)   # position index in [0, Ro)
    pid_c  = tl.program_id(1)   # channel block

    Ro = 124                                # hardcoded (avg-pool + slice)
    L  = 249                               # hardcoded (in_3 input length)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # ------------------------------------------------------------------
    # 1.  Fused conv1d + GELU  (stride=2, pad=15, dilation=1, groups=16)
    # conv_out_len = (249 + 30 - 48 + 1)/2 + 1 = 136
    # Two input positions per output pool row: i0 = 2*r, i1 = 2*r+1
    # ------------------------------------------------------------------
    conv_val = tl.zeros([BLOCK_C], dtype=tl.float32)

    for k in range(K):
        in3_dil = 2 * k - 15              # dilation*kernel + padding_offset
        in3_pad = tl.where(in3_dil >= 0, in3_dil, 0)

        w_val = tl.load(
            weight_ptr + c_offs * K + k,
            mask=c_mask, other=0.0
        ).to(tl.float32)

        for i in range(2):
            in3_i     = (2 * pid_r + i) + in3_pad
            in3_valid = in3_i >= 0
            in3_idx   = in3_i * C + c_offs
            in3_val   = tl.load(
                in3_ptr + in3_idx,
                mask=c_mask & in3_valid,
                other=0.0
            ).to(tl.float32)
            conv_val = conv_val + in3_val * w_val

    conv_bias_val = tl.load(conv_bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    conv_val = conv_val + conv_bias_val

    # ------------------------------------------------------------------
    # 2. GELU: x * 0.5 * (1 + tanh(sqrt(2/π)*(x + 0.044773*x^3)))
    # ------------------------------------------------------------------
    INV_SQRT2PI = 0.7978845608028654
    MAX_VAL     = 89.0    # stabilisation: clamp y >= -MAX_VAL so exp(2y) fits
    y           = INV_SQRT2PI * (conv_val + 0.044715 * (conv_val * conv_val * conv_val))
    y_stacked   = tl.maximum(y, -MAX_VAL)
    e2y         = tl.exp(y_stacked * 2.0)
    gelu_val    = 0.5 * conv_val * (1.0 + (e2y - 1.0) / (e2y + 1.0))

    # ------------------------------------------------------------------
    # 3. Average pooling of two input positions  → GELU
    # pair index = 2 * pid_r
    # ------------------------------------------------------------------
    pool_idx0 = 2 * pid_r
    in3_idx0  = pool_idx0 * C + c_offs
    in3_idx1  = (pool_idx0 + 1) * C + c_offs
    in3_v0    = tl.load(in3_ptr + in3_idx0, mask=c_mask, other=0.0).to(tl.float32)
    in3_v1    = tl.load(in3_ptr + in3_idx1, mask=c_mask, other=0.0).to(tl.float32)
    pool_val  = (in3_v0 + in3_v1) * 0.5

    x3_pool   = pool_val * pool_val * pool_val
    # Compute stabilized tanh once, reuse for both numerator and denominator
    pool_exp  = tl.exp(2.0 * INV_SQRT2PI * (pool_val + 0.044715 * x3_pool) - (-MAX_VAL))
    gelu_pool = 0.5 * pool_val * (pool_exp - 1.0) / (pool_exp + 1.0)

    # ------------------------------------------------------------------
    # 4. Layer norm  (normalize over C)
    # ------------------------------------------------------------------
    add_val = gelu_val + gelu_pool
    mean    = tl.sum(add_val, axis=0) / C
    diff    = add_val - mean
    var     = tl.sum(diff * diff, axis=0) / C
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    safe_offs = tl.where(c_mask, c_offs, 0)
    ln_w      = tl.load(ln_weight_ptr + safe_offs, mask=c_mask, other=1.0).to(tl.float32)
    ln_b      = tl.load(ln_bias_ptr   + safe_offs, mask=c_mask, other=0.0).to(tl.float32)
    result    = (diff * inv_std * ln_w + ln_b).to(out_ptr.dtype.element_ty)

    # ------------------------------------------------------------------
    # 5. Write output[0, Ro, c]  shape [1, Ro, C]
    # ------------------------------------------------------------------
    tl.store(out_ptr + Ro * C + c_offs, result, mask=c_mask)


@torch.fx.wrap
def fused_conv_gelu_pool_ln(in_3, in_4, in_2, in_1, in_0):
    """
    Fused: conv1d+gelu+avgpool+slice+add+transpose+layernorm+dropout(train=False)
    Args:
        in_3  – input activations  [1, 768, 249]
        in_4  – conv weight        [768, 48, 31]
        in_2  – conv bias          [768]
        in_1  – layernorm weight   [768]
        in_0  – layernorm bias     [768]
    Returns:
        output [1, Ro, C]  with Ro=124, C=768
    """
    C  = 768
    Ro = 124
    BLOCK_C = 256

    out = torch.empty((1, Ro, C), dtype=in_3.dtype, device=in_3.device)

    fused_conv_gelu_pool_add_ln_kernel[(Ro, triton.cdiv(C, BLOCK_C))](
        in_3, in_4, in_2, in_1, in_0, out,
        C=C, K=48, BLOCK_C=BLOCK_C,
    )

    return (out,)


# ------------------------------------------------------------------
# Pattern / replacement interface
# ------------------------------------------------------------------

def pattern(x, weight, conv_bias, ln_weight, ln_bias):
    conv1d = torch.conv1d(x, weight, conv_bias, (2,), (15,), (1,), 16)
    tmp_4  = torch.nn.functional.gelu(conv1d)
    tmp_5  = torch.avg_pool1d(x, (2,), (2,), (0,), False, True)
    tmp_6  = tmp_5[(Ellipsis, slice(None, 124, None))]
    tmp_7  = tmp_4[(Ellipsis, slice(None, 124, None))]
    tmp_8  = tmp_6 + tmp_7
    tmp_9  = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), ln_weight, ln_bias, 1e-05)
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
    return (tmp_11,)


def replacement_args(x, weight, conv_bias, ln_weight, ln_bias):
    return (x, weight, conv_bias, ln_weight, ln_bias)


def replacement_func():
    return fused_conv_gelu_pool_ln