import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Strategy: match sum(dim=-1) + unsqueeze(-1) [2-op linear subgraph].
# Replace with a single sum(dim=-1, keepdim=True) call — one fewer PyTorch
# dispatch than the original two-op chain, and correct output shape so the
# downstream div(in_0, sums) correctly normalizes in_0.
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@torch.fx.wrap
def fused_row_sum_keepdim(in_0):
    """Single fused sum(keepdim=True) — equivalent to sum + unsqueeze
    but avoids the extra unsqueeze dispatch."""
    return in_0.sum(dim=-1, keepdim=True)


def replacement_func():
    return fused_row_sum_keepdim