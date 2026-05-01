"""
Fuse: in_5.to(float32) -> torch.tensor(1.0) -> (1.0 - x) -> to(bool) -> masked_fill(-inf)
into a single Triton kernel.

Logic per element x (int64):
  val = 1.0 - float(x)
  result = -3.4028234663852886e+38 if val != 0.0 else 0.0
  i.e. result = 0.0 when x==1, else -inf
"""
import torch
import triton
import triton.language as tl

_NEG_INF = -3.4028234663852886e+38


@triton.jit
def attn_mask_fuse_kernel(
    in5_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load int64 values
    x = tl.load(in5_ptr + offs, mask=mask, other=1)  # other=1 => output 0.0

    # Fused: val = 1.0 - float(x);  result = (val != 0) ? -inf : 0.0
    val = 1.0 - x.to(tl.float32)
    result = tl.where(val != 0.0, tl.full([BLOCK_SIZE], _NEG_INF, tl.float32), val)

    tl.store(out_ptr + offs, result, mask=mask)


@torch.fx.wrap
def fused_attn_mask(in_5):
    n = in_5.numel()
    out = torch.empty(in_5.shape, dtype=torch.float32, device=in_5.device)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    attn_mask_fuse_kernel[grid](in_5, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Pattern: mirrors the Dynamo-compiled graph.
# torch.tensor(1.0, dtype=torch.float32) is constant-folded by Dynamo into the
# scalar 1.0, so the subtraction node becomes call_function(operator.sub, 1.0, x).
# Using `1.0 - tmp_4` in the pattern produces the same call_function form.
# ---------------------------------------------------------------------------
def pattern(in_5):
    tmp_4 = in_5.to(torch.float32)
    tmp_6 = 1.0 - tmp_4
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8


def replacement_args(in_5):
    return (in_5,)


def replacement_func():
    return fused_attn_mask