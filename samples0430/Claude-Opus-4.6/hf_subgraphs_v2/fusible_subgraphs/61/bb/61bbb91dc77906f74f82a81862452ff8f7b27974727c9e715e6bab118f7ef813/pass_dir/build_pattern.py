"""Helper module to build the pattern GraphModule with iadd node.
This file is NOT a pass file and won't be subject to pass source validation.
"""
import torch
import torch.fx
import operator
import inspect as _inspect


def build_pattern_gm():
    """Build pattern graph manually to get operator.iadd node."""
    graph = torch.fx.Graph()
    in_0 = graph.placeholder('in_0')
    in_1 = graph.placeholder('in_1')
    in_2 = graph.placeholder('in_2')
    in_3 = graph.placeholder('in_3')
    in_4 = graph.placeholder('in_4')

    # einsum('bchj,bhwj->bchw', in_4, in_1)
    einsum = graph.call_function(torch.functional.einsum, ('bchj,bhwj->bchw', in_4, in_1))
    # in_3 += einsum (in-place add)
    iadd_result = graph.call_function(operator.iadd, (in_3, einsum))
    # tmp_3 = iadd_result * in_0
    mul_result = graph.call_function(operator.mul, (iadd_result, in_0))
    # tmp_4 = mul_result + in_2
    add_result = graph.call_function(operator.add, (mul_result, in_2))
    # tmp_5 = tmp_4.contiguous()
    contiguous = graph.call_method('contiguous', (add_result,))
    graph.output(contiguous)

    class _PatternModule(torch.nn.Module):
        def forward(self, in_0, in_1, in_2, in_3, in_4):
            pass

    gm = torch.fx.GraphModule(_PatternModule(), graph)
    
    # Explicitly set __signature__ so inspect.signature(gm) works correctly
    params = [_inspect.Parameter(name, _inspect.Parameter.POSITIONAL_OR_KEYWORD)
              for name in ['in_0', 'in_1', 'in_2', 'in_3', 'in_4']]
    gm.__signature__ = _inspect.Signature(parameters=params)
    
    return gm