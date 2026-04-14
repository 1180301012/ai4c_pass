"""
Pass: FuseAddLayerNorm_ret_ln_sum
Fuses   tmp_2 = x + y
        tmp_4 = layer_norm(tmp_2, (1024,), weight, bias, 1e-5)
        return (tmp_4, tmp_2)
into a single Triton kernel that computes both outputs in one pass.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused Triton kernel: elementwise add + layer-norm
# OUTPUT_FP16: True → store as fp16, False → store as bfloat16
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['N_COLS', 'OUTPUT_FP16'],
)
@triton.jit
def _fused_add_ln_ret_ln_sum_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    SUM_ptr, OUT_ptr,
    N_COLS,
    eps,
    OUTPUT_FP16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row (of length N_COLS).
      s   = x + y           → stored to SUM_ptr
      out = layer_norm(s)*W+B → stored to OUT_ptr
    All arithmetic in float32; stores cast back to the original dtype.
    """
    row = tl.program_id(0)
    row_offset = row * N_COLS

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    # Load inputs, upcast to f32
    x = tl.load(X_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + y

    # Layer-norm: mean
    mean = tl.sum(s, axis=0) / N_COLS

    # Variance
    diff = s - mean
    var  = tl.sum(diff * diff, axis=0) / N_COLS

    # Normalize
    inv_std    = 1.0 / tl.sqrt(var + eps)
    normalized = diff * inv_std

    # Scale + shift
    w   = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b   = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = normalized * w + b

    # Store with explicit cast back to original dtype
    if OUTPUT_FP16:
        tl.store(SUM_ptr + row_offset + cols, s.to(tl.float16),   mask=mask)
        tl.store(OUT_ptr + row_offset + cols, out.to(tl.float16), mask=mask)
    else:
        tl.store(SUM_ptr + row_offset + cols, s.to(tl.bfloat16),   mask=mask)
        tl.store(OUT_ptr + row_offset + cols, out.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (decorated with @torch.fx.wrap so FX treats it as a leaf)
# Returns (ln_out, sum_out)  ←→  pattern output order (tmp_4, tmp_2)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_add_ln_ret_ln_sum(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [1024]
    in_1 : weight [1024]
    in_2 : x      [B, S, 1024]   (fp16 or bf16)
    in_3 : y      [B, S, 1024]   (fp16 or bf16)
    Returns: (ln_out, sum_out) — both in the same dtype as inputs
    """
    N_COLS      = in_2.shape[-1]
    N_ROWS      = in_2.numel() // N_COLS
    OUTPUT_FP16 = (in_2.dtype == torch.float16)

    sum_out = torch.empty_like(in_2)
    ln_out  = torch.empty_like(in_2)

    _fused_add_ln_ret_ln_sum_kernel[(N_ROWS,)](
        in_2, in_3,        # X, Y
        in_1, in_0,        # W (weight), B (bias)
        sum_out, ln_out,
        N_COLS=N_COLS,
        eps=1e-5,
        OUTPUT_FP16=OUTPUT_FP16,
    )

    return (ln_out, sum_out)


# ---------------------------------------------------------------------------
# Pattern, replacement_args, replacement_func
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return (tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_ln_ret_ln_sum