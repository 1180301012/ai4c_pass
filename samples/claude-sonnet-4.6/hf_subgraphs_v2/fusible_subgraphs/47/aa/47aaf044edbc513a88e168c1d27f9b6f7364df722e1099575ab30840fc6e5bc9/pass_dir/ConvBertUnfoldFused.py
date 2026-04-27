import torch
import triton
import triton.language as tl

# ── Try multiple anchor strategies to find what's in the _decomposed graph.
#    Pattern A: high-level method-call form (transpose as anchor via call_method)
#    Pattern B: ATen permute.default (in case transpose.int decomposes to permute)
#
#    We use the SHORT pattern (no reshape dims) so one pass handles C=16 and C=384.


def pattern(in_0):
    # Use method calls (standard torch.fx: both target and pattern trace through
    # torch.nn.functional.unfold identically, producing the same im2col node)
    tmp_1 = in_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


# ─── Triton kernel ───────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['C', 'L'],
)
@triton.jit
def _unfold_fused_kernel(
    input_ptr,
    output_ptr,
    C,
    L,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flat index f = l*C*9 + c*9 + w  →  input[0, c, l-4+w]
    Output shape: [1, L, C*9]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    w = offsets % 9
    c = (offsets // 9) % C
    l = offsets // (9 * C)

    input_pos = l - 4 + w
    valid = (input_pos >= 0) & (input_pos < L) & mask

    input_offset = c * L + input_pos
    x = tl.load(input_ptr + input_offset, mask=valid, other=0.0)
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def _unfold_fused(in_0):
    x = in_0.contiguous()
    C = x.shape[1]
    L = x.shape[2]
    total_elements = L * C * 9
    output = torch.empty((1, L, C * 9), dtype=x.dtype, device=x.device)
    _unfold_fused_kernel[
        lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    ](x, output, C, L, total_elements)
    return output


def replacement_func():
    return _unfold_fused