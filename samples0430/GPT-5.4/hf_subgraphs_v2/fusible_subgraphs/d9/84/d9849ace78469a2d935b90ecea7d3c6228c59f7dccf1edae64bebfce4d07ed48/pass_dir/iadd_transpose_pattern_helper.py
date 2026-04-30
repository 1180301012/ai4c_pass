import operator
import torch


_pattern_graph = torch.fx.Graph()
_pattern_in_0 = _pattern_graph.placeholder("in_0")
_pattern_in_1 = _pattern_graph.placeholder("in_1")
_pattern_in_2 = _pattern_graph.call_function(operator.iadd, args=(_pattern_in_1, _pattern_in_0))
_pattern_graph.output(_pattern_in_2)

pattern = torch.fx.GraphModule(torch.nn.Module(), _pattern_graph)