import torch
import triton
import triton.language as tl


# ── Pattern: gelu + reshape + reshape  (confirmed to match) ───────────────────
# Our kernel returns [1,248,768] → framework's pad adds the zero row.

def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# ── Triton GELU kernel ─────────────────────────────────────────────────────────
# Input:  [1, 124, 1536]  flat 190 464 elements
# Output: [1, 248, 768]  flat 191 232 elements  (same elements, different shape)
# The framework's pad then runs to produce [1, 249, 768].
#
# 1-D flat grid, BLOCK_SIZE=4096 → only 47 blocks for 190 464 elements.

@triton.jit
def _gelu_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x   = tl.load(in_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    y32 = 0.5 * x32 * (1.0 + tl.math.erf(x32 * 0.7071067811865476))
    tl.store(out_ptr + offs, y32.to(x.dtype), mask=mask)


@torch.fx.wrap
def triton_gelu_reshape(in_0):
    # in_0: [1, 124, 1536]  →  out: [1, 248, 768]  (gelu result, reshaped)
    N = in_0.numel()           # 190 464
    BLOCK_SIZE = 4096
    NUM_BLOCKS = (N + BLOCK_SIZE - 1) // BLOCK_SIZE  # 47

    # Must return [1, 248, 768] — framework's reshape+pad views are free
    out = torch.empty((1, 248, 768), dtype=in_0.dtype, device=in_0.device)
    _gelu_kernel[(NUM_BLOCKS,)](in_0, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return triton_gelu_reshape