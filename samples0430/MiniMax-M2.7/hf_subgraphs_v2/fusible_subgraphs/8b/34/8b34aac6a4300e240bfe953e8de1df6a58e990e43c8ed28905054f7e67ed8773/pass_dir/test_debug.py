#!/usr/bin/env python3
import torch
import torch.fx

# Test pattern matching with PyTorch's internal tools
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9

# Try to trace the pattern
print("Attempting to trace pattern...")
try:
    traced_pattern = torch.fx.symbolic_trace(pattern)
    print("Pattern traced successfully!")
    print("\nPattern nodes:")
    for node in traced_pattern.graph.nodes:
        print(f"  {node.name}: {node.op} {node.target}")
except Exception as e:
    print(f"Failed to trace pattern: {e}")
    import traceback
    traceback.print_exc()

# Test with a class
class GraphModule(torch.nn.Module):
    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.conv2d(in_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
        tmp_4 = torch.sigmoid(tmp_3)
        tmp_5 = in_3 * tmp_4
        tmp_6 = torch.sigmoid(tmp_2)
        tmp_7 = in_2 * tmp_6
        tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
        tmp_9 = tmp_5 + tmp_8
        return (tmp_9,)

print("\nAttempting to trace model...")
try:
    model = GraphModule()
    traced_model = torch.fx.symbolic_trace(model)
    print("Model traced successfully!")
    print("\nModel nodes:")
    for node in traced_model.graph.nodes:
        print(f"  {node.name}: {node.op} {node.target}")
except Exception as e:
    print(f"Failed to trace model: {e}")
    import traceback
    traceback.print_exc()