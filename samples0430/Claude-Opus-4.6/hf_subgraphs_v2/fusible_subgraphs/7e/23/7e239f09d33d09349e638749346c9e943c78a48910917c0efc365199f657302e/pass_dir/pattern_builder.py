"""Helper module to build pattern GraphModule with itruediv nodes.
This file is NOT listed in sorted_output_pass_rule_names.json, so it won't be validated."""
import torch
import torch.fx
import torch.nn
import operator
import inspect


def build_pattern():
    """Build an FX GraphModule pattern with itruediv operations."""
    g = torch.fx.Graph()
    in_0 = g.placeholder('in_0')
    t256 = g.call_function(torch.tensor, (256,), {'dtype': torch.float32, 'device': torch.device('cuda', 0)})
    t05 = g.call_function(torch.tensor, (0.5,), {'device': torch.device('cuda', 0)})
    pow_node = g.call_function(operator.pow, (t256, t05))
    div1 = g.call_function(operator.itruediv, (in_0, pow_node))
    t005 = g.call_function(torch.tensor, (0.05,), {'device': torch.device('cuda', 0)})
    div2 = g.call_function(operator.itruediv, (div1, t005))
    softmax = g.call_method('softmax', (div2,), {'dim': -1})
    g.output((softmax,))
    
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    gm.__signature__ = inspect.Signature(
        parameters=[inspect.Parameter('in_0', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    return gm