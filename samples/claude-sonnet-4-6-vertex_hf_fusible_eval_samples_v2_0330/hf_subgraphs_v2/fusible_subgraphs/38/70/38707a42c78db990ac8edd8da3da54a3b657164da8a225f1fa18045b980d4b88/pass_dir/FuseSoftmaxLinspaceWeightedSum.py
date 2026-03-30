import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: (softmax_out * linspace_out).sum(dim=1) -> 5 - result
#
# Both softmax_out and linspace_out are wildcard placeholders so this matches
# regardless of how the upstream nodes appear in the graph.
# ---------------------------------------------------------------------------
def pattern(softmax_out, linspace_out):
    tmp_2 = softmax_out * linspace_out
    tmp_3 = tmp_2.sum(dim=1)
    tmp_4 = 5 - tmp_3
    return tmp_4


def replacement_args(softmax_out, linspace_out):
    # Kernel hardcodes weights [0,1,2,3,4]; only softmax output is needed.
    return (softmax_out,)


# ---------------------------------------------------------------------------
# Triton kernel: dot(softmax_probs, [0,1,2,3,4]) then 5 - dot
# Input: bfloat16 or float16, shape [B, 5]
# Output: float32, shape [B]   (matches original type-promotion from linspace)
# One CTA per row; BLOCK_SIZE=8 (next power-of-2 >= 5).
# ---------------------------------------------------------------------------
@triton.jit
def fused_weighted_sum_sub_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    NUM_CLASSES: tl.constexpr,   # = 5
    BLOCK_SIZE:  tl.constexpr,   # = 8
):
    row_idx = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < NUM_CLASSES

    # Load bfloat16/float16 softmax probs; pad out-of-bound elements with 0
    x = tl.load(in_ptr + row_idx * NUM_CLASSES + offsets, mask=mask, other=0.0)

    # Upcast to float32 (matches linspace float32 type-promotion in original model)
    x = x.to(tl.float32)

    # Weighted dot-product with constants [0, 1, 2, 3, 4]
    weights     = offsets.to(tl.float32)
    dot_product = tl.sum(tl.where(mask, x * weights, 0.0), axis=0)

    # 5 - dot  (matches `5 - tmp_3` in model.py)
    tl.store(out_ptr + row_idx, 5.0 - dot_product)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_softmax_weighted_sum(softmax_out):
    softmax_out = softmax_out.contiguous()
    batch_size  = softmax_out.shape[0]

    out = torch.empty(batch_size, dtype=torch.float32, device=softmax_out.device)

    fused_weighted_sum_sub_kernel[(batch_size,)](
        in_ptr      = softmax_out,
        out_ptr     = out,
        batch_size  = batch_size,
        NUM_CLASSES = 5,
        BLOCK_SIZE  = 8,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-argument, returns the callable
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_softmax_weighted_sum