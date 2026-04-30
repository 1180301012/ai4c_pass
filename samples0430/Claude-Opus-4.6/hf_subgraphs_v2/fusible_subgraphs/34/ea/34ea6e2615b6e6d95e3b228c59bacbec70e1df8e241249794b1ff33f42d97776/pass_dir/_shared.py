import torch
import triton
import triton.language as tl


@triton.jit
def add_flatten_transpose_kernel(
    a_ptr, b_ptr, out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fuses add + flatten + transpose: reads NCHW, writes contiguous NLC."""
    s = tl.program_id(0)  # spatial position index
    c_offsets = tl.arange(0, BLOCK_C)
    mask = c_offsets < C

    # Read from NCHW layout: offset for (0, c, s) = c * HW + s
    input_offsets = c_offsets * HW + s
    a_vals = tl.load(a_ptr + input_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + input_offsets, mask=mask, other=0.0)

    # Add
    result = a_vals + b_vals

    # Write contiguous NLC: offset for (0, s, c) = s * C + c
    out_offsets = s * C + c_offsets
    tl.store(out_ptr + out_offsets, result, mask=mask)


def custom_dispatch(arg0, arg1, route):
    """Dispatch function shared by all passes."""
    route_val = int(route)
    if route_val == 0:
        return _add_flatten_transpose(arg0, arg1)
    return arg0


def _add_flatten_transpose(conv_out, in_4):
    """Add two NCHW tensors and output as contiguous NLC."""
    C = conv_out.shape[1]
    H = conv_out.shape[2]
    W = conv_out.shape[3]
    HW = H * W

    out = torch.empty((1, HW, C), dtype=conv_out.dtype, device=conv_out.device)

    # Choose BLOCK_C as next power of 2
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    add_flatten_transpose_kernel[(HW,)](
        conv_out, in_4, out,
        C=C,
        HW=HW,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out