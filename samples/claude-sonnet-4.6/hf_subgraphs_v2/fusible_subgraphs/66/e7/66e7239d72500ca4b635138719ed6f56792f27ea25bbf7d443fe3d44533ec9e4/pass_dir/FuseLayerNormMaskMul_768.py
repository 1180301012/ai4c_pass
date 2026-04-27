import torch
import triton
import triton.language as tl


@triton.jit
def fused_layernorm_mask_kernel(
    in3_ptr,        # input: [B, S, 768] fp16/bf16
    w_ptr,          # LN weight: [768] fp16/bf16
    b_ptr,          # LN bias:   [768] fp16/bf16
    in0_ptr,        # attention mask: [B*S] int64
    out_ln_ptr,     # tmp_4: [B, S, 768] fp16/bf16
    out_mask_ptr,   # tmp_7: [B, S, 768] fp32
    out_masked_ptr, # tmp_8: [B, S, 768] fp32
    BLOCK_SIZE: tl.constexpr,
):
    # N is fixed at 768 for this kernel
    N = 768

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    valid = cols < N

    # ------------------------------------------------------------------
    # Load input row in native dtype (fp16/bf16), promote to fp32
    # ------------------------------------------------------------------
    x = tl.load(in3_ptr + row * N + cols, mask=valid, other=0.0)
    x_f32 = x.to(tl.float32)

    # ------------------------------------------------------------------
    # Layer norm: mean over N valid elements
    # ------------------------------------------------------------------
    mean = tl.sum(x_f32, 0) / N

    # ------------------------------------------------------------------
    # Layer norm: variance (zero out the BLOCK_SIZE-N padding positions)
    # ------------------------------------------------------------------
    xm = tl.where(valid, x_f32 - mean, 0.0)
    var = tl.sum(xm * xm, 0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-12)

    # Normalised values
    x_hat = xm * rstd

    # ------------------------------------------------------------------
    # Affine transform using LN weight / bias (native dtype → fp32)
    # ------------------------------------------------------------------
    w = tl.load(w_ptr + cols, mask=valid, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=valid, other=0.0).to(tl.float32)
    y = x_hat * w + b   # fp32

    # ------------------------------------------------------------------
    # Attention-mask value for this row: int64 → fp32 scalar
    # ------------------------------------------------------------------
    mask_val = tl.load(in0_ptr + row).to(tl.float32)

    # Broadcast mask scalar to full row (fp32)
    ones = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    mask_row = ones * mask_val

    # ------------------------------------------------------------------
    # tmp_4: y rounded to the native input dtype (fp16 or bf16)
    # tmp_8: simulates PyTorch type-promotion fp16 * fp32 = fp32
    #        We round-trip y through the native dtype, then multiply.
    # x.dtype is known at Triton compile time from the pointer type.
    # ------------------------------------------------------------------
    y_native  = y.to(x.dtype)             # fp32 → fp16/bf16  (tmp_4)
    y_for_mul = y_native.to(tl.float32)   # fp16/bf16 → fp32
    out_masked = y_for_mul * mask_val     # fp32               (tmp_8)

    # ------------------------------------------------------------------
    # Store outputs (only to valid column positions)
    # ------------------------------------------------------------------
    tl.store(out_ln_ptr     + row * N + cols, y_native,   mask=valid)  # tmp_4
    tl.store(out_mask_ptr   + row * N + cols, mask_row,   mask=valid)  # tmp_7
    tl.store(out_masked_ptr + row * N + cols, out_masked, mask=valid)  # tmp_8


# ---------------------------------------------------------------------------
# Pattern: mirrors model.py exactly — positional args, same ops, same order
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(tmp_4)
    tmp_7 = tmp_6.float()
    tmp_8 = tmp_4 * tmp_7
    return (tmp_7, tmp_8, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def fused_layernorm_mask(in_0, in_1, in_2, in_3):
    """
    Fused layer-norm + attention-mask broadcast + elementwise multiply.

    in_0 : [B, S]      int64      attention mask
    in_1 : [768]       fp16/bf16  LN bias
    in_2 : [768]       fp16/bf16  LN weight
    in_3 : [B, S, 768] fp16/bf16  hidden states

    Returns (tmp_7, tmp_8, tmp_4) matching the original model output order.
    """
    total_rows = in_3.shape[0] * in_3.shape[1]  # B * S

    # Allocate outputs
    out_ln     = torch.empty_like(in_3)                       # tmp_4  native dtype
    out_mask   = torch.empty_like(in_3, dtype=torch.float32)  # tmp_7  fp32
    out_masked = torch.empty_like(in_3, dtype=torch.float32)  # tmp_8  fp32

    fused_layernorm_mask_kernel[(total_rows,)](
        in_3,
        in_2,        # LN weight
        in_1,        # LN bias
        in_0,        # attention mask
        out_ln,
        out_mask,
        out_masked,
        BLOCK_SIZE=1024,
        num_warps=4,
    )

    # Return order: (tmp_7, tmp_8, tmp_4)
    return out_mask, out_masked, out_ln


def replacement_func():
    return fused_layernorm_mask