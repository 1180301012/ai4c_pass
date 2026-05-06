import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused element-wise add + layer-norm
# One program handles one row of HIDDEN elements.
# Dropout(training=False) is identity, so it is elided.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'HIDDEN': 256}, num_warps=2),
        triton.Config({'HIDDEN': 256}, num_warps=4),
        triton.Config({'HIDDEN': 256}, num_warps=8),
    ],
    key=['HIDDEN'],
)
@triton.jit
def fused_add_ln_kernel(
    x_ptr,       # [N_ROWS, HIDDEN]  first addend
    y_ptr,       # [N_ROWS, HIDDEN]  second addend
    w_ptr,       # [HIDDEN]          layer-norm weight
    b_ptr,       # [HIDDEN]          layer-norm bias
    out_ptr,     # [N_ROWS, HIDDEN]  output
    HIDDEN: tl.constexpr,
    eps,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, HIDDEN)

    # Load and add (upcast to fp32 for numerical stability)
    x = tl.load(x_ptr + row * HIDDEN + offs).to(tl.float32)
    y = tl.load(y_ptr + row * HIDDEN + offs).to(tl.float32)
    s = x + y

    # Online layer-norm: mean then variance
    mean = tl.sum(s, 0) / HIDDEN
    d    = s - mean
    var  = tl.sum(d * d, 0) / HIDDEN
    norm = d / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)
    out = norm * w + b

    # Store with original dtype
    if IS_BF16:
        tl.store(out_ptr + row * HIDDEN + offs, out.to(tl.bfloat16))
    elif IS_FP16:
        tl.store(out_ptr + row * HIDDEN + offs, out.to(tl.float16))
    else:
        tl.store(out_ptr + row * HIDDEN + offs, out)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_ln(x, y, w, b):
    """
    x : [*, HIDDEN]    first operand to add (token embeddings scaled by 16)
    y : [*, HIDDEN]    second operand to add (position embeddings)
    w : [HIDDEN]       layer-norm weight (gamma)
    b : [HIDDEN]       layer-norm bias   (beta)
    Returns fused(x + y) with layer-norm applied, same dtype as x.
    Dropout(training=False) is identity and absorbed.
    """
    HIDDEN     = x.shape[-1]
    out        = torch.empty_like(x)
    N          = x.numel() // HIDDEN
    IS_BF16    = x.dtype == torch.bfloat16
    IS_FP16    = x.dtype == torch.float16

    fused_add_ln_kernel[(N,)](
        x, y, w, b, out,
        HIDDEN=HIDDEN,
        eps=1e-5,
        IS_BF16=IS_BF16,
        IS_FP16=IS_FP16,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(tmp_5, tmp_9, in_3, in_2):
    """
    Matches:  add(x, y)  →  layer_norm(sum, (256,), w, b, 1e-05)  →  dropout(..., training=False)
    In the target graph:
        x = tmp_5 = token_embed * 16.0   (tmp_4 * 16.0)
        y = tmp_9 = pos_embed at index 2  (from arange(0,1).expand(1,-1)+2)
        w = in_3 = layer-norm weight
        b = in_2 = layer-norm bias
    """
    tmp_10 = tmp_5 + tmp_9
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_3, in_2, 1e-05)
    tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.1, training=False)
    return tmp_12


def replacement_args(tmp_5, tmp_9, in_3, in_2):
    return (tmp_5, tmp_9, in_3, in_2)


def replacement_func():
    return fused_add_ln