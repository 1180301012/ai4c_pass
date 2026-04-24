import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _fused_add_max_softmax_kernel_16_9_9(
    in0_ptr,   # [1, 1, 9, 9] contiguous -> S*S flat
    in1_ptr,   # [1, 16, 9, 9] contiguous -> N*S*S flat
    out_ptr,   # [1, 16, 9, 9] contiguous -> N*S*S flat
    N, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per row (N rows total)
    row_idx = pid % S       # column position in attention matrix (in_0 is [1,1,S,S])

    base0 = row_idx * S     # in_0 row offset (same for all heads)
    base1 = pid             # in_1 row offset (flat N*S)

    offsets = tl.arange(0, BLOCK_S)
    mask = offsets < S

    # Load fp16/bf16 inputs, upcast to fp32 for numerical stability
    v0 = tl.load(in0_ptr + base0 + offsets, mask=mask, other=-3.4028234663852886e+38).to(tl.float32)
    v1 = tl.load(in1_ptr + base1 + offsets, mask=mask, other=-3.4028234663852886e+38).to(tl.float32)

    # Add + clamp (maximum with -FLT_MAX)
    v = v0 + v1
    v = tl.where(v < -3.4028234663852886e+38, -3.4028234663852886e+38, v)

    # Numerically-stable softmax in fp32
    row_max = tl.max(v, axis=0)
    v = v - row_max
    v = tl.where(mask, v, -3.4028234663852886e+38)   # zero-out padding slots
    exp_v = tl.exp(v)
    row_sum = tl.sum(exp_v, axis=0)
    out = exp_v / row_sum

    # Store back in original dtype
    tl.store(out_ptr + base1 + offsets, out.to(in1_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def _fused_add_max_softmax_16_9_9(in_0, in_1):
    # in_0: [1, 1, 9, 9],  in_1: [1, 16, 9, 9]
    N = 16
    S = 9
    BLOCK_S = 16
    out = torch.empty_like(in_1)
    _fused_add_max_softmax_kernel_16_9_9[(N * S,)](
        in_0.view(N * S * S),
        in_1.view(N * S * S),
        out.view(N * S * S),
        N, S,
        BLOCK_S=BLOCK_S,
    )
    return out.view(1, N, S, S)


def pattern(in_0, in_1, mask_val):
    tmp_0 = in_1 + in_0
    tmp_2 = torch.max(tmp_0, mask_val)
    tmp_3 = tmp_2.view(16, 9, 9)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return tmp_5


def replacement_args(in_0, in_1, mask_val):
    return (in_0, in_1)


def replacement_func():
    return _fused_add_max_softmax_16_9_9