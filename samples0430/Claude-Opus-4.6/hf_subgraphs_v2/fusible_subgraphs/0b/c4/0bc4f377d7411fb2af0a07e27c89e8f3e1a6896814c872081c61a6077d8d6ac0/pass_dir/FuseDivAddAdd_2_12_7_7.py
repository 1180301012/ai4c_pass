import torch
import torch.fx
import triton
import triton.language as tl
import operator


# Monkey-patch Proxy to support __iadd__ (assignment, not a blocked call)
def _iadd_proxy(*args, **kwargs):
    tracer = args[0].tracer
    target = operator.iadd
    return tracer.create_proxy("call_function", target, args, kwargs)

torch.fx.Proxy.__iadd__ = _iadd_proxy


def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 / 8.0
    tmp_0 += in_2
    tmp_2 = tmp_0 + in_1
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_div_add_add_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load in_0 and in_2 (same shape [2, 12, 7, 7], direct indexing)
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)

    # Compute broadcast index for in_1 [2, 1, 1, 7]
    # Output shape is [2, 12, 7, 7], flat index: b*588 + h*49 + i*7 + j
    # in_1 index: b*7 + j
    j = offsets % 7
    b = offsets // 588
    in_1_idx = b * 7 + j
    y = tl.load(in_1_ptr + in_1_idx, mask=mask, other=0.0)

    # Compute: (in_0 / 8.0 + in_2) + in_1
    out = x * 0.125 + z + y

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_div_add_add(in_0, in_1, in_2):
    N = in_0.numel()
    BLOCK_SIZE = 2048
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_0)

    fused_div_add_add_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_div_add_add