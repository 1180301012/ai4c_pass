import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: iadd -> float() -> softmax(dim=-1) -> type_as -> dropout(p=0.1, training=False)
# All five ops are fused into a single Triton kernel.
# dropout with training=False is an identity, so it is dropped in the kernel.
# ---------------------------------------------------------------------------


def pattern(in_0, in_1):
    in_1 += in_0
    tmp_1 = in_1.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(in_1)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return (tmp_4,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel
# One GPU program handles one row (the softmax dimension is the last axis).
# BLOCK_SIZE is set to the next power-of-two >= n_cols at kernel-launch time.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["n_cols"],
)
@triton.jit
def _fused_add_softmax_kernel(
    in0_ptr,   # pointer to in_0 (original, unmodified)
    in1_ptr,   # pointer to in_1 (original, unmodified)
    out_ptr,   # pointer to output (same dtype as in_1)
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    base = row_idx * n_cols

    # Load both inputs (native dtype → float32 for numerics)
    x0 = tl.load(in0_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in1_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

    # Element-wise add (equivalent to iadd)
    x = x0 + x1

    # Set out-of-bounds lanes to -inf so they don't corrupt max/sum
    x = tl.where(mask, x, float("-inf"))

    # Numerically-stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.exp(x)
    x_sum = tl.sum(x, axis=0)
    x = x / x_sum

    # Zero out the lanes that were out-of-bounds (they already are 0 via exp(-inf)/sum,
    # but being explicit avoids any edge-case NaN from 0/0 if n_cols==0).
    x = tl.where(mask, x, 0.0)

    # Store – Triton casts float32 → out_ptr element type (fp16/bf16/fp32) automatically
    tl.store(out_ptr + base + offsets, x, mask=mask)


# ---------------------------------------------------------------------------
# Host-side wrapper (must be @torch.fx.wrap so FX does not trace into it)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    """
    Computes softmax(in_0 + in_1, dim=-1) in the original dtype of in_1,
    fusing the iadd / float() / softmax / type_as / no-op dropout.
    """
    # Ensure contiguous (no-op if already contiguous)
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()

    n_cols = in_1.shape[-1]
    n_rows = in_1.numel() // n_cols

    out = torch.empty_like(in_1)

    # Choose smallest power-of-two that covers the whole row
    BLOCK_SIZE = max(triton.next_power_of_2(n_cols), 16)

    _fused_add_softmax_kernel[(n_rows,)](
        in_0, in_1, out,
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out,)


# ---------------------------------------------------------------------------
# Required by the AI4C pass framework
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_add_softmax