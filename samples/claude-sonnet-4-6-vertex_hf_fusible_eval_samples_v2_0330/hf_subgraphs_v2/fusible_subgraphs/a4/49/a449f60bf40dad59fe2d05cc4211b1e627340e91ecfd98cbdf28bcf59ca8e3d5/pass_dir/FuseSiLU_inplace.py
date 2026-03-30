import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: single detach call — matched 3 times in the graph.
# detach is the only operation matchable via the pattern tracer for this graph.
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_1 = in_0.detach()
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Replacement: identity passthrough.
# detach() is a value no-op for inference; returning x directly avoids the
# detach metadata work and produces bitwise-identical numeric values.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def detach_identity(x):
    return x


def replacement_func():
    return detach_identity