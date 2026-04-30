import torch
import torch.fx as fx

# Simple test to understand pattern matching

class TestModule(torch.nn.Module):
    def forward(self, in_0, in_1, in_2):
        tmp_1 = in_2 * in_1
        tmp_2 = tmp_1 + in_0
        unbind = torch.unbind(tmp_2, dim=2)
        tmp_4 = unbind[0]
        tmp_5 = unbind[1]
        tmp_6 = tmp_5.permute(0, 2, 1)
        return (tmp_6, tmp_4)


def pattern(in_0, in_1, in_2):
    mul = in_2 * in_1
    add = mul + in_0
    parts = torch.unbind(add, dim=2)
    first = parts[0]
    second = parts[1]
    permuted = second.permute(0, 2, 1)
    return permuted, first


# Trace the model and pattern
model = TestModule()
traced_model = fx.symbolic_trace(model)
traced_pattern = fx.symbolic_trace(pattern)

print("Model graph:")
for node in traced_model.graph.nodes:
    print(f"  {node.op}: {node.target} ({node.name})")

print("\nPattern graph:")
for node in traced_pattern.graph.nodes:
    print(f"  {node.op}: {node.target} ({node.name})")