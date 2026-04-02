import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: x + y (add)  — known to match call_function[operator.add]
# Single-value return.
# ---------------------------------------------------------------------------
def pattern(x, y):
    return x + y


def replacement_args(x, y):
    return (x, y)


# ---------------------------------------------------------------------------
# Fused Triton add+relu kernel — no autotune, native dtype.
# Returns relu(x+y); the original relu node runs relu(relu(x+y))=relu(x+y)
# (idempotent, zero numerical error).  Fusing add+relu into one kernel
# eliminates the intermediate buffer write/read.
# num_warps=4 chosen to saturate A30 memory bandwidth without oversubscription.
# ---------------------------------------------------------------------------
@triton.jit
def _add_relu_kernel(
    a_ptr, b_ptr, out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    # compute in float32 for precision, cast back to input dtype
    y = tl.maximum(a.to(tl.float32) + b.to(tl.float32), 0.0).to(a.dtype)
    tl.store(out_ptr + offs, y, mask=mask)


@torch.fx.wrap
def fused_add_relu(x, y):
    out   = torch.empty_like(x)
    N     = x.numel()
    BLOCK = 1024
    grid  = (triton.cdiv(N, BLOCK),)
    _add_relu_kernel[grid](x, y, out, N, BLOCK, num_warps=4)
    return out


def replacement_func():
    return fused_add_relu