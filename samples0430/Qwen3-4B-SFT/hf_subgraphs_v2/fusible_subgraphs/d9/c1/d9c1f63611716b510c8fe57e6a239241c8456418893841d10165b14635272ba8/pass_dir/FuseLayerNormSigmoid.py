import torch
import triton
import triton.language as tl


# ── pattern ───────────────────────────────────────────────────────────────────
# Match ONLY the layer_norm call (single output: tmp_9).
# y.sigmoid() is LEFT OUTSIDE the pattern so that pattern returns one value only.
def pattern(x, w, b):
    """Match: layer_norm(x, (256,), w, b, 1e-05)"""
    return torch.nn.functional.layer_norm(x, (256,), w, b, 1e-05)


def replacement_args(x, w, b):
    return (x, w, b)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Fixed num_warps=4 → 128 threads per CTA (32 logical-elements per thread).
# More CTAs/SM (= smaller blocks) gives higher GPU occupancy & latency hiding
# than num_warps=8 or 16 for this small workload (N_ROWS=300, BLOCK=256).
@triton.jit
def _layernorm_kernel(
    x_ptr,    # [N_ROWS, 256]  – input
    w_ptr,    # [256]          – LN weight
    b_ptr,    # [256]          – LN bias
    out_ptr,  # [N_ROWS, 256]  – output
    N_ROWS,
    stride_rows,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    row_off = row * stride_rows
    cols = tl.arange(0, BLOCK)

    # float32 accumulation for numerical accuracy
    x = tl.load(x_ptr + row_off + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    b = tl.load(b_ptr + cols).to(tl.float32)

    mean = tl.sum(x, axis=0) / BLOCK
    x_c  = x - mean
    var  = tl.sum(x_c * x_c, axis=0) / BLOCK
    rstd = tl.rsqrt(var + tl.constexpr(1e-5))   # eps as constexpr → stays fp32

    out = x_c * rstd * w + b

    tl.store(out_ptr + row_off + cols, out.to(x.dtype))


# ── wrapper ───────────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_layernorm_256(x, w, b):
    """
    x : [..., 256]  – input tensor; last dim must be 256
    w : [256]       – LN weight
    b : [256]       – LN bias
    Returns ln(x) cast back to x.dtype.
    """
    N_ROWS     = x.numel() // 256
    stride_rows = x.stride(-2)   # batch stride equals 256 for [B,1,256] inputs
    BLOCK      = 256             # constexpr: matches N_COLS == 256

    out = torch.empty_like(x)

    # Fixed num_warps=8 → 8 warps × 32 threads = 256 physical threads
    # per CTA → 1 element per thread → fewer serial reduction steps
    _layernorm_kernel[(N_ROWS,)](
        x, w, b,
        out,
        N_ROWS,
        stride_rows,
        BLOCK=BLOCK,
        num_warps=8,
    )

    return out


# ── replacement hook ──────────────────────────────────────────────────────────
def replacement_func():
    return triton_layernorm_256