#!/usr/bin/env python3
import torch
import torch.fx
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

# Test pattern
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.nn.functional.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    tmp_6 = torch.nn.functional.sigmoid(tmp_2)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9

# Test with a class model
class GraphModule(torch.nn.Module):
    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
        tmp_4 = torch.nn.functional.sigmoid(tmp_3)
        tmp_3 = None
        tmp_5 = in_3 * tmp_4
        tmp_4 = None
        tmp_6 = torch.nn.functional.sigmoid(tmp_2)
        tmp_2 = None
        tmp_7 = in_2 * tmp_6
        tmp_6 = None
        tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
        tmp_7 = None
        tmp_9 = tmp_5 + tmp_8
        tmp_5 = tmp_8 = None
        return (tmp_9,)

# Try to trace using standard FX
print("Attempting to trace pattern...")
try:
    traced_pattern = torch.fx.symbolic_trace(pattern)
    print("Pattern traced successfully!")
    print("\nPattern graph:")
    traced_pattern.graph.print_tabular()
except Exception as e:
    print(f"Failed to trace pattern: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)

# Trace the model
print("Attempting to trace model...")
try:
    model = GraphModule()
    traced_model = torch.fx.symbolic_trace(model)
    print("Model traced successfully!")
    print("\nModel graph:")
    traced_model.graph.print_tabular()
except Exception as e:
    print(f"Failed to trace model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)

# Try matching
print("Attempting to match pattern against model...")
try:
    matcher = SubgraphMatcher(traced_pattern.graph, match_output=False, match_placeholder=False)
    matches = matcher.match(traced_model.graph)
    print(f"Found {len(matches)} matches")
    for i, match in enumerate(matches):
        print(f"  Match {i}: placeholder_map={match.placeholder_map}")
except Exception as e:
    print(f"Failed to match: {e}")
    import traceback
    traceback.print_exc()