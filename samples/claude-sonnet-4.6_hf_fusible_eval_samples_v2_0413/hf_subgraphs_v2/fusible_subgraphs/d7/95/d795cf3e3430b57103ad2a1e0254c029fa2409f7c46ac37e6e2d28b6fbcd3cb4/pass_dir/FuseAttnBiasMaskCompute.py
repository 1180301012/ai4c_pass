import torch
import triton
import triton.language as tl


@triton.jit
def ne_maskedfill_kernel(
    inp_ptr,
    out_ptr,
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ne + masked_fill(-1000.0):
      out[i] = -1000.0  if inp[i] != 0
             = inp[i]   otherwise  (inp[i] == 0.0 → out[i] = 0.0)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    val = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    result = tl.where(val != 0.0, -1000.0, val)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_ne_maskedfill(tmp_12):
    """
    Single-output replacement for:  (tmp_12 != 0) → masked_fill(-1000.0)
    tmp_12 shape: (1, 361, 49, 49)  float32
    """
    total = tmp_12.numel()          # 865 801
    out   = torch.empty_like(tmp_12)

    BLOCK_SIZE = 1024
    n_programs = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    ne_maskedfill_kernel[(n_programs,)](
        tmp_12, out, total,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern:  tmp_12 is taken as input (sub is outside this subgraph).
# Single output → no tuple-return FX rewriter issue.
# tmp_12 remains in the graph; downstream `== 0` and `masked_fill(0.0)` still
# use it correctly (and the second masked_fill is a no-op on our output).
# ---------------------------------------------------------------------------
def pattern(tmp_12):
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    return tmp_14


def replacement_args(tmp_12):
    return (tmp_12,)


def replacement_func():
    return fused_ne_maskedfill