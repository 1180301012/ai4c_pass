import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: elementwise multiply  out[i] = x[i] * y[i]
# Replaces only the *gated-mul* step; the upstream silu stays as a native
# PyTorch op.  Fixed BLOCK_SIZE=1024 → 257 blocks, ~2.4 blocks/SM on A30.
# ---------------------------------------------------------------------------

@triton.jit
def mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * y, mask=mask)


@torch.fx.wrap
def fused_silu_mul_dropout(x, y):
    """Replace the gated-mul: x is already the silu output."""
    N    = x.numel()
    BLOCK = 1024
    out  = torch.empty_like(x)
    grid = ((N + BLOCK - 1) // BLOCK,)
    mul_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK, num_warps=4)
    return out


# ---------------------------------------------------------------------------
# Pass interface
#
# Pattern: any element-wise *.  replacement_args returns the mul's two
# inputs UNCHANGED (no graph navigation) so the silu node stays live and
# the Triton kernel receives (silu_output, in_1), computing silu_out * in_1
# which is exactly the original computation.
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    return in_0 * in_1


def replacement_args(in_0, in_1):
    # Return the mul's inputs as-is; no graph surgery needed because
    # the silu node is still referenced as in_0 → it is NOT dead code.
    return (in_0, in_1)


def replacement_func():
    return fused_silu_mul_dropout