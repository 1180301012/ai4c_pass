import torch
import triton
import triton.language as tl

# ── Critical: wrap F.unfold so it appears as a single leaf node when torch.fx
# traces the pattern function (mirrors the custom tracer used to create model.py).
from torch.nn.functional import unfold
torch.fx.wrap(unfold)

from pass_dir._shared_dispatch import convbert_dispatch, _REGISTRY


# ─── Pattern ────────────────────────────────────────────────────────────────────
def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0, "route_16_8")


# ─── Triton kernel for C=16, group_size=8 ───────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['total_elements'],
)
@triton.jit
def _unfold_kernel_16_8(
    input_ptr,
    output_ptr,
    C,
    L,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flat output index f = l*C*9 + c*9 + k
      k   = f % 9
      c   = (f // 9) % C
      l   = f // (9 * C)
      input_pos = l - 4 + k   (padding=4 for kernel_size=9)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    k = offsets % 9
    c = (offsets // 9) % C
    l = offsets // (9 * C)

    input_pos = l - 4 + k
    valid = (input_pos >= 0) & (input_pos < L) & mask

    input_offset = c * L + input_pos
    x = tl.load(input_ptr + input_offset, mask=valid, other=0.0)
    tl.store(output_ptr + offsets, x, mask=mask)


def _run_route_16_8(in_0):
    x = in_0.contiguous()
    C = x.shape[1]
    L = x.shape[2]
    group_size = 8
    num_groups = C // group_size
    total_rows = L * num_groups
    total_elements = total_rows * group_size * 9

    output = torch.empty((total_rows, group_size, 9), dtype=x.dtype, device=x.device)
    _unfold_kernel_16_8[
        lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    ](x, output, C, L, total_elements)
    return output


# ─── Register in shared registry ────────────────────────────────────────────────
_REGISTRY["route_16_8"] = _run_route_16_8


def replacement_func():
    return convbert_dispatch