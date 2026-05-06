import torch
import triton
import triton.language as tl


# ── pattern ───────────────────────────────────────────────────────────────────
# Use descriptive, non-`_in_N` names throughout to avoid framework name-collision
# issues (where arg names like `_in_1` conflict with model-placeholder names).
#
# a          = tmp_11  – sigmoid(LN(linear_out))   [free pass] computed name
# y          = tmp_9   – input gate (in_9 sigmoid) [free pass] computed name
# v_fixed_10 = in_10   – input_out                 [free pass] computed name
# v_fixed_11 = in_11   – param_out                 [free pass] computed name
# w_fixed_1  = in_5    – LN(param_out) weight (model input in_5)
# b_fixed_1  = in_4    – LN(param_out) bias  (model input in_4)
# w_fixed_0  = in_1    – LN(input_out)  weight (model input in_1)
# b_fixed_0  = in_0    – LN(input_out)  bias  (model input in_0)
# We avoid any name containing `_in_1`, `_in_0`, `_in_N` that appear
# simultaneously in the argument list AND the function body.
def pattern(a, y, v_fixed_10, v_fixed_11, w_fixed_1, b_fixed_1, w_fixed_0, b_fixed_0):
    # a:  sigmoid( LN(linear_out) )
    # y:  in_9  (input gate sigmoid, separate computation outside pattern)
    # v_fixed_10/in_10: input to LN(in_10)
    # w_fixed_1/in_5:   LN(in_11) weight/bias (model input placeholders)
    # w_fixed_0/in_1:   LN(in_10) weight/bias (model input placeholders)
    # b_fixed_1/in_4:   LN(in_11) bias
    # b_fixed_0/in_0:   LN(in_10) bias
    tmp_12  = torch.nn.functional.layer_norm(v_fixed_11, (256,), w_fixed_1, b_fixed_1, 1e-05)
    tmp_14  = tmp_12.unsqueeze(-2)
    tmp_15  = a * tmp_14
    tmp_13  = torch.nn.functional.layer_norm(v_fixed_10, (256,), w_fixed_0, b_fixed_0, 1e-05)
    tmp_16  = y.sigmoid() * tmp_13
    tmp_17  = tmp_15 + tmp_16
    return tmp_17


def replacement_args(a, y, v_fixed_10, v_fixed_11, w_fixed_1, b_fixed_1, w_fixed_0, b_fixed_0):
    # Map pattern freed-argument names → model actual-node names:
    #   a          → LN(linear_out) output,  used as: sigmoid(
    #   y          → in_9 (input gate),       used as: sigmoid(in_9)
    #   v_fixed_10 → in_10, goes to LN(in_10)
    #   v_fixed_11 → in_11, LN weight/bias are q_fixed_1/b_fixed_1
    #   w_fixed_0  → in_1,   LN weight/bias are w_fixed_0/b_fixed_0
    return (a, y, v_fixed_10, v_fixed_11, w_fixed_0, b_fixed_0, w_fixed_1, b_fixed_1)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# All 4 array loads happen before any arithmetic so the GPU can pipeline them.
# LN mean1 and mean2 share the same register space → no extra DRAM traffic.
@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4, "num_stages": 1}),
        triton.Config({"num_warps": 8, "num_stages": 1}),
        triton.Config({"num_warps": 16, "num_stages": 1}),
        triton.Config({"num_warps": 4, "num_stages": 2}),
        triton.Config({"num_warps": 8, "num_stages": 2}),
        triton.Config({"num_warps": 16, "num_stages": 2}),
    ],
    key=["N_ROWS"],
)
@triton.jit
def _fused_elemwise_kernel(
    a_ptr,    # sigmoid( LN(linear_out) )  [N_ROWS, N_COLS]
    y9_ptr,   # in_9 (gate input)          [N_ROWS, N_COLS]
    x10_ptr,  # in_10                       [N_ROWS, N_COLS]
    x11_ptr,  # in_11                       [N_ROWS, N_COLS]
    w10_ptr,  # in_1  = LN(in_10) weight    [N_COLS]
    b10_ptr,  # in_0  = LN(in_10) bias      [N_COLS]
    w1_ptr,   # in_5  = LN(in_11) weight    [N_COLS]
    b1_ptr,   # in_4  = LN(in_11) bias      [N_COLS]
    out_ptr,  # output                          [N_ROWS, N_COLS]
    N_ROWS, stride_rows, eps,
    BLOCK: tl.constexpr,
):
    row     = tl.program_id(0)
    row_off = row * stride_rows
    cols    = tl.arange(0, BLOCK)

    # Issue all 4 large-row loads before any arithmetic (hide memory latency)
    a    = tl.load(a_ptr  + row_off + cols).to(tl.float32)
    y9   = tl.load(y9_ptr + row_off + cols).to(tl.float32)
    x10  = tl.load(x10_ptr + row_off + cols).to(tl.float32)
    x11  = tl.load(x11_ptr + row_off + cols).to(tl.float32)

    # LN(in_11)
    mean1  = tl.sum(x11, axis=0) / BLOCK
    x11_c  = x11 - mean1
    var1   = tl.sum(x11_c * x11_c, axis=0) / BLOCK
    rstd1  = tl.rsqrt(var1 + eps)
    w1     = tl.load(w1_ptr  + cols).to(tl.float32)
    b1     = tl.load(b1_ptr  + cols).to(tl.float32)
    ln11   = x11_c * rstd1 * w1 + b1

    # LN(in_10)
    mean2  = tl.sum(x10, axis=0) / BLOCK
    x10_c  = x10 - mean2
    var2   = tl.sum(x10_c * x10_c, axis=0) / BLOCK
    rstd2  = tl.rsqrt(var2 + eps)
    w2     = tl.load(w10_ptr + cols).to(tl.float32)
    b2     = tl.load(b10_ptr + cols).to(tl.float32)
    ln10   = x10_c * rstd2 * w2 + b2

    # sigmoid(in_9)
    sig_y9 = 1.0 / (1.0 + tl.exp(-y9))

    result = a * ln11 + sig_y9 * ln10
    tl.store(out_ptr + row_off + cols, result.to(a.dtype))


# ── wrapper ───────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_elemwise(a, y, v_fixed_10, v_fixed_11, w_fixed_0, b_fixed_0, w_fixed_1, b_fixed_1):
    """
    a          : sigmoid( LN(linear_out) )  shape [B, 1, 256]
    y          : in_9  (input gate)         shape [B, 1, 256]
    v_fixed_10 : in_10                      shape [B, 1, 256]
    v_fixed_11 : in_11                      shape [B, 256]
    w_fixed_0  : in_1                        [256]  – LN(in_10) weight
    b_fixed_0  : in_0                        [256]  – LN(in_10) bias
    w_fixed_1  : in_5                        [256]  – LN(in_11) weight
    b_fixed_1  : in_4                        [256]  – LN(in_11) bias
    Returns     : tmp_17 of shape [B, 1, 256]
    """
    N_ROWS     = a.numel() // 256
    stride_rows = a.stride(-2)
    eps        = 1e-5
    BLOCK      = 256

    out = torch.empty_like(a)

    _fused_elemwise_kernel[(N_ROWS,)](
        a, y,
        v_fixed_10,             # x10
        v_fixed_11,             # x11
        w_fixed_0,  b_fixed_0,  # w2, b2  = w_fixed_0=in_1, b_fixed_0=in_0 → LN(in_10)
        w_fixed_1, b_fixed_1,   # w1, b1  = w_fixed_1=in_5, b_fixed_1=in_4 → LN(in_11)
        out,
        N_ROWS,
        stride_rows,
        eps,
        BLOCK=BLOCK,
    )

    return out


# ── replacement hook ──────────────────────────────────────────────────────────
def replacement_func():
    return fused_elemwise