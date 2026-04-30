import torch
import torch.fx
import torch.nn.functional as F
import operator


def make_silu_add_pattern():
    g = torch.fx.Graph()
    in_0 = g.placeholder('in_0')
    in_1 = g.placeholder('in_1')
    silu = g.call_function(F.silu, args=(in_1,), kwargs={'inplace': True})
    add = g.call_function(operator.add, args=(silu, in_0))
    g.output(add)
    return torch.fx.GraphModule(torch.nn.Module(), g)