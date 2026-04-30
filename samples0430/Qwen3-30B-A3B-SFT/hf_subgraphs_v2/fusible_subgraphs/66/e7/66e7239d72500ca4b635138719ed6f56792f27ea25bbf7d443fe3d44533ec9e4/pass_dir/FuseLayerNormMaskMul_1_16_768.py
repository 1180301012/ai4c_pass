import torch
import triton
import triton.language as tl


def pattern(in_1, in_2, in_3):
    """
    Matches:  tmp_4 = layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    in_1 = bias [768], in_2 = weight [768], in_3 = input [1,16,768]
    """
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    return tmp_4


def replacement_args(in_1, in_2, in_3):
    # in_1: [768]   fp16/bf16  (ln_bias)
    # in_2: [768]   fp16/bf16  (ln_weight)
    # in_3: [1, 16, 768] fp16/bf16  (input)
    return (in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel: layer-norm  (one row per program, BLOCK_SIZE >= hidden)
# ---------------------------------------------------------------------------

@triton.jit
def _ln_kernel(
    in_1_ptr,   # [H] fp16/bf16 – LN bias
    in_2_ptr,   # [H] fp16/bf16 – LN weight
    in_3_ptr,   # [S, H] fp16/bf16 – input (row-major)
    out_ptr,    # [S, H] fp16/bf16 – output
    S,
    hidden,
    eps,
    BLOCK_SIZE: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    row_id  = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < hidden

    # Load input row in fp32
    x = tl.load(in_3_ptr + row_id * hidden + offsets,
                mask=mask, other=0.0).to(tl.float32)

    # Mean (masked-out positions hold 0 and don't pollute the sum)
    mean = tl.sum(x, axis=0) / hidden

    # Variance – zero out out-of-range lanes before squaring
    x_c  = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(x_c * x_c, axis=0) / hidden

    rstd   = tl.rsqrt(var + eps)
    x_norm = x_c * rstd

    # Affine transform
    weight = tl.load(in_2_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias   = tl.load(in_1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out    = x_norm * weight + bias

    # Store in the original dtype
    if IS_FP16:
        tl.store(out_ptr + row_id * hidden + offsets, out.to(tl.float16), mask=mask)
    else:
        tl.store(out_ptr + row_id * hidden + offsets, out.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_layer_norm(in_1, in_2, in_3):
    """
    in_1 : [H]           fp16/bf16 – LN bias
    in_2 : [H]           fp16/bf16 – LN weight
    in_3 : [1, S, H]     fp16/bf16 – input tensor
    returns tmp_4 [1, S, H]
    """
    S    = in_3.shape[1]
    H    = in_3.shape[2]
    IS_FP16 = (in_3.dtype == torch.float16)

    BLOCK_SIZE = 1024  # next power-of-2 >= 768

    out = torch.empty_like(in_3)

    _ln_kernel[(S,)](
        in_1, in_2, in_3, out,
        S, H,
        1e-12,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_FP16=IS_FP16,
        num_warps=8,
    )

    return out


def replacement_func():
    return triton_layer_norm