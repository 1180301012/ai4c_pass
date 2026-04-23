import operator
import torch


def build_iadd_relu_pattern():
    graph = torch.fx.Graph()
    acc = graph.placeholder("acc")
    in_2 = graph.placeholder("in_2")
    add_out = graph.call_function(operator.iadd, (acc, in_2))
    relu_out = graph.call_function(torch.nn.functional.relu, (add_out,), {"inplace": True})
    graph.output(relu_out)
    return torch.fx.GraphModule(torch.nn.Module(), graph)