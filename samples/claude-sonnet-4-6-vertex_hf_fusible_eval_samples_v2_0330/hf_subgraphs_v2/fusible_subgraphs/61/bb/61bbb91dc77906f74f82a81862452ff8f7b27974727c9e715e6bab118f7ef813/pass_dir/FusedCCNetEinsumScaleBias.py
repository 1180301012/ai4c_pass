import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match  x * scalar + bias + contiguous
#
# In the model graph this corresponds to the TAIL of the subgraph:
#   in_5 = in_3 after iadd (in_3 + einsum)
#   tmp_3 = in_5 * in_0    ← x * scalar
#   tmp_4 = tmp_3 + in_2   ← result + bias
#   tmp_5 = tmp_4.contiguous()
#
# We skip the einsum + iadd and only fuse the cheaper element-wise part.
# This avoids the iadd pattern-matching issue and still fuses 2 element-wise
# ops + contiguous into a single Triton kernel (halving memory traffic).
# ---------------------------------------------------------------------------
def pattern(x, in_0, in_2):
    tmp_3 = x * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


def replacement_args(x, in_0, in_2):
    return (x, in_0, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fuses  out = x * scalar + bias  (replaces mul + add + contiguous)
#
# x, bias: [B, C, H, W] contiguous tensors
# scalar:  shape=[]  (0-d scalar tensor)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_scale_bias_kernel(
    x_ptr, scalar_ptr, bias_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements

    x    = tl.load(x_ptr    + offsets, mask=mask, other=0.0)
    s    = tl.load(scalar_ptr)                              # scalar tensor []
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    out  = x * s + bias
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_scale_bias(x, in_0, in_2):
    """
    Fused replacement for:  tmp_5 = (x * in_0 + in_2).contiguous()
    Single Triton kernel eliminates one extra memory round-trip vs two ops.
    """
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (
        (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
    )
    fused_scale_bias_kernel[grid](
        x, in_0, in_2, out,
        n_elements,
    )
    return out


def replacement_func():
    return fused_scale_bias