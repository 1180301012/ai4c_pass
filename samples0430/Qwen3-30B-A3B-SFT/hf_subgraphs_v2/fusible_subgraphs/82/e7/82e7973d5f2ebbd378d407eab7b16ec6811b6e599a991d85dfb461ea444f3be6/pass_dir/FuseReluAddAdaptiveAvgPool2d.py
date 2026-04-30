import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Minimal diagnostic pattern: just add (no relu, no pooling).
# Tests whether operator.add nodes exist in the model graph.
# If "No specific node failures" → model has no operator.add nodes (aten level)
# If "MatchFailure"         → model has operator.add nodes
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    return in_0 + in_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _add_kernel(
    in0_ptr, in1_ptr, out_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x0 = tl.load(in0_ptr + offs, mask=mask, other=0.0)
    x1 = tl.load(in1_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x0 + x1, mask=mask)


@torch.fx.wrap
def fused_relu_add_avgpool(in_0, in_1):
    N = in_0.numel()
    out = torch.empty_like(in_0)
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _add_kernel[grid](in_0, in_1, out, N, BLOCK=BLOCK)
    return out


def replacement_func():
    return fused_relu_add_avgpool