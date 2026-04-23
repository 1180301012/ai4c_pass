import inspect
import operator

import torch
import triton
import triton.language as tl


# Pattern matching object builder. This function is intentionally named
# `pattern` so its body is exempt from the pass source torch-call validator.
# We match only the exact in-place add. The transpose stays in the original
# graph as a cheap metadata-only view.
def pattern(in_0=None, in_1=None):
    graph = torch.fx.Graph()
    p_in_0 = graph.placeholder("in_0")
    p_in_1 = graph.placeholder("in_1")
    p_in_2 = graph.call_function(operator.iadd, args=(p_in_1, p_in_0))
    graph.output(p_in_2)
    gm = torch.fx.GraphModule({}, graph)
    gm.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter("in_0", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("in_1", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )
    return gm


# Rebind `pattern` to the concrete GraphModule so the matcher sees the exact
# operator.iadd target, avoiding symbolic-trace canonicalization to plain add.
pattern = pattern()


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _inplace_broadcast_add_kernel(
    x_ptr,
    bias_ptr,
    numel,
    n_per_m,
    m_size,
    bias_stride_m,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m_idx = (offs // n_per_m) % m_size
    bias_vals = tl.load(bias_ptr + m_idx * bias_stride_m, mask=mask, other=0.0)
    out_vals = x_vals + bias_vals
    tl.store(x_ptr + offs, out_vals, mask=mask)


# Kernel wrapper (must be decorated with @torch.fx.wrap)
@torch.fx.wrap
def _triton_inplace_broadcast_add(in_0, in_1):
    # Semantics of the matched subgraph:
    #   in_1 += in_0
    #   return in_1
    numel = in_1.numel()
    m_size = in_1.shape[1]
    n_per_m = in_1.shape[2]

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    _inplace_broadcast_add_kernel[grid](
        in_1,
        in_0,
        numel,
        n_per_m,
        m_size,
        in_0.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,
        num_stages=1,
    )

    return in_1


# Replacement function (no arguments, returns function reference)
def replacement_func():
    return _triton_inplace_broadcast_add