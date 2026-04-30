import torch
import triton
import triton.language as tl


def pattern(in_5):
    tmp_10 = in_5[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    return tmp_12


def replacement_args(in_5):
    return (in_5,)


@triton.jit
def value_expand_kernel(
    val_ptr, out_ptr,
    SEQ_LEN: tl.constexpr,
    DIM: tl.constexpr,
    N_HEADS: tl.constexpr,
):
    pid = tl.program_id(0)  # one per seq position
    row_offset = pid * DIM
    offsets = tl.arange(0, DIM)

    # Load value for this seq position
    val = tl.load(val_ptr + row_offset + offsets)

    # Store to all heads
    for h in range(N_HEADS):
        tl.store(out_ptr + h * SEQ_LEN * DIM + pid * DIM + offsets, val)


@torch.fx.wrap
def fused_value_expand(in_5):
    out = torch.empty(1, 8, 3, 256, dtype=in_5.dtype, device=in_5.device)

    value_expand_kernel[(3,)](
        in_5, out,
        SEQ_LEN=3,
        DIM=256,
        N_HEADS=8,
        num_warps=1,
    )

    return out


def replacement_func():
    return fused_value_expand