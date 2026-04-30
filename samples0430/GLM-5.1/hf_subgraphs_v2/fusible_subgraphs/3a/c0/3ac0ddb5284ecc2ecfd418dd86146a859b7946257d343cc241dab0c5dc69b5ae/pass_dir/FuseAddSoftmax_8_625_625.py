import torch
import triton
import triton.language as tl

from pass_dir._shared_dispatch import fused_add_softmax_dispatch


def pattern(in_0, in_1):
    """Pattern matching for float32 variant with view dimensions (8, 625, 625).
    
    Matches the computation:
    add(in_1, in_0) -> view(8,625,625) -> softmax(dim=-1) -> view(1,8,625,625) -> view(8,625,625) -> dropout(p=0.0,training=False)
    
    Returns (tmp_5, tmp_3) where both are views of the softmax result.
    """
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 625, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 625, 625)
    tmp_4 = tmp_3.view(8, 625, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
    return (tmp_5, tmp_3)


def replacement_args(in_0, in_1):
    """Extract arguments and append route string for dispatch."""
    return (in_0, in_1, "route_8_625_625")


def replacement_func():
    """Return the shared dispatch wrapper (same across all passes)."""
    return fused_add_softmax_dispatch