import torch


def pattern(in_0, in_1):
    """
    Match the exact pattern from BAAI_AltCLIP.
    """
    tmp_2 = in_1[slice(None, None, None), slice(None, 7, None)]
    tmp_3 = tmp_2.expand(2, 7)
    tmp_4 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1):
    """Extract arguments."""
    return (in_0, in_1)


@torch.fx.wrap
def optimized(in_0, in_1):
    # Inline operations to avoid intermediate nodes
    return (
        in_1[slice(None, None, None), slice(None, 7, None)].expand(2, 7),
        in_0[slice(None, None, None), None, None, slice(None, None, None)]
    )


def replacement_func():
    """
    Return optimized function.
    Use exact same syntax as pattern to avoid graph structure mismatch.
    """
    return optimized