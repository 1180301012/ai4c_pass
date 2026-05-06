import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Match: view(1,32,15,15) -> bicubic interpolate to (15,15) -> flatten(2) -> transpose(1,2)
    Since input and output sizes are identical, this entire chain produces x.shape unchanged.
    Replacing it with a simple stride-cast kernel eliminates expensive bicubic interpolation.
    """
    v = x.view(1, 32, 15, 15)
    y = torch.nn.functional.interpolate(v, size=(15, 15), mode='bicubic', align_corners=False)
    z = y.flatten(2)
    w = z.transpose(1, 2)
    return w


def replacement_args(x):
    return (x,)


@triton.jit
def noop_bicubic_copy_kernel(
    src_ptr,
    out_ptr,
    C: tl.constexpr,
    FLATTENED_SIZE: tl.constexpr,   # H*W = 225
):
    """
    Reads src[0, :, d] (CHW layout, batch=1) and writes to out[i, d].
    Since the bicubic returns the same spatial size, the output is a
    stride-copy: out[i, ch] = src[0, i, ch] for i in [0, FLATTENED_SIZE).
    Input strides: [C*FLATTENED_SIZE, FLATTENED_SIZE, 1]
    Output strides: [FLATTENED_SIZE*C, C, 1]
    """
    pid = tl.program_id(0)          # row index in [0, FLATTENED_SIZE)
    d = tl.arange(0, C)             # channel axis [0 .. C-1]
    src_offs = d + pid * C          # src[0, pid, d]: [C]
    dst_offs = pid * C + d          # out[pid, d]:   [C]
    vals = tl.load(src_ptr + src_offs)
    tl.store(out_ptr + dst_offs, vals)


@torch.fx.wrap
def noop_bicubic_1x32x15x15(x):
    """
    Replaces the bicubic-upsample chain with a straight stride-copy.
    Output shape: [1, 225, 32]  (same total elements as input [1,32,15,15]).
    """
    C = 32
    FLATTENED_SIZE = 225
    out = torch.empty((1, FLATTENED_SIZE, C), dtype=x.dtype, device=x.device)
    noop_bicubic_copy_kernel[(FLATTENED_SIZE,)](
        x, out,
        C=C,
        FLATTENED_SIZE=FLATTENED_SIZE,
    )
    return out


def replacement_func():
    return noop_bicubic_1x32x15x15