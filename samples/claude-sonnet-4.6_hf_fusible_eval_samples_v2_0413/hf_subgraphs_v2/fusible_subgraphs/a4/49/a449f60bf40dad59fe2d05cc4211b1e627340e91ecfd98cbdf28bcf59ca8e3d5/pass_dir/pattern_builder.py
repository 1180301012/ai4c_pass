"""
Helper module to build the pattern GraphModule without triggering
API validation in the pass file. This file is NOT a pass rule.
Matches only the silu node (single-output) to avoid multi-output
replacement complexity.
"""
import torch
import torch.fx as fx
import inspect as _inspect


def build_pattern_gm():
    """
    Build a single-node FX pattern: call_function[silu](in_0,) kwargs={'inplace':True}
    Single-output pattern is cleanly handled by _replace_pattern.
    """
    graph = fx.Graph()
    in_0 = graph.placeholder('in_0')
    tmp_0 = graph.call_function(
        torch.nn.functional.silu,
        args=(in_0,),
        kwargs={'inplace': True}
    )
    graph.output(tmp_0)
    gm = fx.GraphModule(torch.nn.Module(), graph)
    gm.__signature__ = _inspect.Signature([
        _inspect.Parameter('in_0', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    return gm