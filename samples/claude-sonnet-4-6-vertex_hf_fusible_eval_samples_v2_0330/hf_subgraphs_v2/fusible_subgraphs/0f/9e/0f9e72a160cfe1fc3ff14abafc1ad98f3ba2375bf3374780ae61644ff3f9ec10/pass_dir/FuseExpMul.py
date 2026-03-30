import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: fuse scalar exp + element-wise multiply.
# Matches:
#   tmp_5 = in_0.exp()
#   tmp_6 = tmp_5 * tmp_4
# tmp_5 is internal (not returned); tmp_6 is the single observable output.
# ---------------------------------------------------------------------------
def pattern(x, y):
    s = x.exp()
    result = s * y
    return result


# ---------------------------------------------------------------------------
# Triton kernel: loads scalar x, computes exp, multiplies each element of y.
# Computation in float32 for precision; stores back in y's original dtype.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}),
        triton.Config({"BLOCK": 512}),
        triton.Config({"BLOCK": 1024}),
    ],
    key=["N"],
)
@triton.jit
def exp_mul_kernel(
    x_ptr,    # pointer to scalar tensor (logit scale)
    y_ptr,    # pointer to [N] flat tensor
    out_ptr,  # pointer to [N] output
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load scalar and compute exp (reused across all threads in this program)
    x_val = tl.load(x_ptr).to(tl.float32)
    scale = tl.exp(x_val)

    y_raw = tl.load(y_ptr + offs, mask=mask, other=0.0)
    y = y_raw.to(tl.float32)
    out = (scale * y).to(y_raw.dtype)

    tl.store(out_ptr + offs, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_exp_mul(x, y):
    """
    Compute exp(x) * y element-wise.
    x is a scalar tensor; y can be any shape.
    """
    y_flat = y.contiguous().reshape(-1)
    N = y_flat.numel()

    out_flat = torch.empty_like(y_flat)

    BLOCK = min(1024, triton.next_power_of_2(N))
    grid = ((N + BLOCK - 1) // BLOCK,)

    exp_mul_kernel[grid](
        x,
        y_flat,
        out_flat,
        N,
    )

    return out_flat.reshape(y.shape)


# ---------------------------------------------------------------------------
# Required interface
# ---------------------------------------------------------------------------
def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return triton_exp_mul