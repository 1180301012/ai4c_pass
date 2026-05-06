import torch
import triton
import triton.language as tl


# ────────────────────────────────────────────────────────────────────────────
# Pattern: fuse the entire gated-upgate LN group into ONE Triton kernel.
#
# Input args (some are weight/bias tensors):
#   v_a      – linear_out (computed, free pass)
#   v_in_9   – input_gate  (model/input tensor, free pass)
#   v_in_10  – input_out   (model/input tensor, free pass)
#   v_in_11  – param_out   (model/input tensor, free pass)
#   w1/b1    – LN(param_out) weight/bias  [= w5 / b4 model args]
#   w2/b2    – LN(input_out) weight/bias  [= w1  / b0 model args]
#   w3/b3    – LN(linear_out) weight/bias [= w6  / b6 model args]
#
# Outputs (both observable outside the matched subgraph):
#   tmp_11  = sigmoid( LN(v_a) )   ← used by outside code
#   tmp_17  = tmp_11 * unsqueeze(LN(v_in_11)) + sigmoid(v_in_9)*LN(v_in_10)
#
# The framework maps the 12 pattern args to 12 graph nodes via semantic search.
# ────────────────────────────────────────────────────────────────────────────
def pattern(v_a, v_in_9, v_in_10, v_in_11, w1, b1, w2, b2, w3, b3):
    tmp_9  = torch.nn.functional.layer_norm(v_a, (256,), w3, b3, 1e-05)
    tmp_11 = tmp_9.sigmoid()
    tmp_12 = torch.nn.functional.layer_norm(v_in_11, (256,), w1, b1, 1e-05)
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_13 = torch.nn.functional.layer_norm(v_in_10, (256,), w2, b2, 1e-05)
    tmp_16 = v_in_9.sigmoid() * tmp_13
    tmp_17 = tmp_15 + tmp_16
    return tmp_11, tmp_17


def replacement_args(v_a, v_in_9, v_in_10, v_in_11, w1, b1, w2, b2, w3, b3):
    return (v_a, v_in_9, v_in_10, v_in_11, w3, b3, w1, b1, w2, b2)


# ────────────────────────────────────────────────────────────────────────────
# Triton kernel : computes LN(v_a)+sigmoid → tmp_11
#                                    LN(v_10), LN(v_11) → element-wise → tmp_17
# All 4 input tensors have 256 elements per row (batch dim = 300).
# ────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4}),
        triton.Config({"num_warps": 8}),
        triton.Config({"num_warps": 16}),
    ],
    key=["N_ROWS"],
)
@triton.jit
def _ln_sig_elemwise_kernel(
    va_ptr, v9_ptr, v10_ptr, v11_ptr,
    w3_ptr, b3_ptr,    # LN(v_a) weight / bias
    w1_ptr, b1_ptr,    # LN(v_in_11) weight / bias
    w2_ptr, b2_ptr,    # LN(v_in_10) weight / bias
    tmp11_ptr, tmp17_ptr,
    N_ROWS,
    stride_rows,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    row_off = row * stride_rows
    cols = tl.arange(0, BLOCK)

    # ── Load input rows ────────────────────────────────────────────────────────
    va = tl.load(va_ptr + row_off + cols).to(tl.float32)
    v9 = tl.load(v9_ptr + row_off + cols).to(tl.float32)
    v10= tl.load(v10_ptr + row_off + cols).to(tl.float32)
    v11= tl.load(v11_ptr + row_off + cols).to(tl.float32)

    # ── LN(v_a) + sigmoid → tmp_11 ────────────────────────────────────────────
    mean3 = tl.sum(va, axis=0) / BLOCK
    va_c  = va - mean3
    var3  = tl.sum(va_c * va_c, axis=0) / BLOCK
    rstd3 = tl.rsqrt(var3 + eps)
    w3    = tl.load(w3_ptr  + cols).to(tl.float32)
    b3    = tl.load(b3_ptr  + cols).to(tl.float32)
    ln3   = va_c * rstd3 * w3 + b3
    sig_ln3 = 1.0 / (1.0 + tl.exp(-ln3))

    # ── LN(v_in_11) ────────────────────────────────────────────────────────────
    mean1 = tl.sum(v11, axis=0) / BLOCK
    v11_c = v11 - mean1
    var1  = tl.sum(v11_c * v11_c, axis=0) / BLOCK
    rstd1 = tl.rsqrt(var1 + eps)
    w1    = tl.load(w1_ptr  + cols).to(tl.float32)
    b1    = tl.load(b1_ptr  + cols).to(tl.float32)
    ln11  = v11_c * rstd1 * w1 + b1

    # ── LN(v_in_10), and in_9 sigmoid ─────────────────────────────────────────
    mean2 = tl.sum(v10, axis=0) / BLOCK
    v10_c = v10 - mean2
    var2  = tl.sum(v10_c * v10_c, axis=0) / BLOCK
    rstd2 = tl.rsqrt(var2 + eps)
    w2    = tl.load(w2_ptr  + cols).to(tl.float32)
    b2    = tl.load(b2_ptr  + cols).to(tl.float32)
    ln10  = v10_c * rstd2 * w2 + b2
    sig_v9 = 1.0 / (1.0 + tl.exp(-v9))

    # ── tmp_17 = sig_ln3 * ln11 + sig_v9 * ln10 ──────────────────────────────
    tmp17 = sig_ln3 * ln11 + sig_v9 * ln10

    # ── Store both outputs ─────────────────────────────────────────────────────
    tl.store(tmp11_ptr + row_off + cols, sig_ln3.to(va.dtype))
    tl.store(tmp17_ptr + row_off + cols, tmp17.to(va.dtype))


# ────────────────────────────────────────────────────────────────────────────
# Wrapper – @torch.fx.wrap makes this opaque to torch.compile's inductor.
# ────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_ln_sig_elem(v_a, v_in_9, v_in_10, v_in_11, w3, b3, w1, b1, w2, b2):
    """
    v_a      : [300, 1, 256]  – linear output
    v_in_9   : [300, 1, 256]  – input_gate
    v_in_10  : [300, 1, 256]  – input_out
    v_in_11  : [300, 256]     – param_out  (unsqueeze applied inside kernel)
    w3/b3    : [256]  – LN(v_a) weight/bias  (matching w6 / b6 model args)
    w1/b1    : [256]  – LN(v_in_11) weight/bias
    w2/b2    : [256]  – LN(v_in_10) weight/bias
    Returns  : (tmp_11 [300,1,256], tmp_17 [300,1,256])
    """
    N_ROWS     = v_a.numel() // 256
    stride_rows = v_a.stride(-2)
    eps        = 1e-5
    BLOCK      = 256

    tmp11 = torch.empty_like(v_a)
    tmp17 = torch.empty_like(v_a)

    _ln_sig_elemwise_kernel[(N_ROWS,)](
        v_a, v_in_9, v_in_10, v_in_11,
        w3, b3,
        w1, b1,
        w2, b2,
        tmp11, tmp17,
        N_ROWS,
        stride_rows,
        eps,
        BLOCK=BLOCK,
    )

    return tmp11, tmp17


def replacement_func():
    return fused_ln_sig_elem