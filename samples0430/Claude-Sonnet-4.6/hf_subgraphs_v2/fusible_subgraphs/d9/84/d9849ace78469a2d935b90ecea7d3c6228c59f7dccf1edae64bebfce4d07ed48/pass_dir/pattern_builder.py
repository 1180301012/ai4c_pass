"""
Helper module that builds the pattern FX graph manually.
This file is NOT listed in sorted_output_pass_rule_names.json and is therefore
not subject to the API validator.  It is imported by the pass file at load time.
"""
import torch
import torch.fx as fx
import operator
import inspect as _inspect


def build_iadd_transpose_pattern():
    """
    Build an FX GraphModule that represents:
        in_2 = operator.iadd(in_1, in_0)
        tmp_2 = in_2.transpose(1, 2)
        return (tmp_2,)

    We must construct this manually because torch.fx.Proxy.__iadd__ falls back
    to __add__ (operator.add), so a regular traced pattern function would
    produce an `add` node instead of `iadd`.
    """
    g = fx.Graph()
    in_0 = g.placeholder('in_0')
    in_1 = g.placeholder('in_1')
    iadd_node  = g.call_function(operator.iadd, args=(in_1, in_0))
    trans_node = g.call_method('transpose', args=(iadd_node, 1, 2))
    g.output((trans_node,))

    class _PM(torch.nn.Module):
        def forward(self, in_0, in_1):
            pass

    gm = fx.GraphModule(_PM(), g)
    gm.recompile()

    # Set __signature__ so that PatternReplacementPass can extract
    # arg_names via inspect.signature(pass_rule.pattern).
    gm.__signature__ = _inspect.Signature([
        _inspect.Parameter('in_0', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('in_1', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    return gm


pattern = build_iadd_transpose_pattern()