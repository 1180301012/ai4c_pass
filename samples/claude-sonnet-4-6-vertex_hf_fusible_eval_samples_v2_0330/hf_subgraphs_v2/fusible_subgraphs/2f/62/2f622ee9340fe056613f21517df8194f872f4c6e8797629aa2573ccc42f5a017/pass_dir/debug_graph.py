"""
Debug script: runs the model through a custom Dynamo backend that prints
the FX graph nodes so we can see the exact op targets.
Run from pass_dir/: python debug_graph.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

# Monkey-patch sym_sum
if not hasattr(torch, 'sym_sum'):
    def _sym_sum(args):
        result = args[0]
        for a in args[1:]:
            result = result + a
        return result
    torch.sym_sum = _sym_sum

# Load the first model
import importlib.util

MODEL_PATH = "../graphs/hf_subgraphs_v2/fusible_subgraphs/bfloat16/3/samples/timm/resnest50d_1s4x24d.in1k/_decomposed/resnest50d_1s4x24d.in1k_start78_end82_3/model.py"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_PATH)

spec = importlib.util.spec_from_file_location("model_debug", MODEL_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
GraphModule = mod.GraphModule

model = GraphModule().cuda().to(torch.bfloat16)

# Create sample inputs
in_0 = torch.tensor(192, dtype=torch.int64, device='cuda')  # scalar int64
in_1 = torch.randn(1, 192, 32, 32, dtype=torch.bfloat16, device='cuda')

captured_gm = [None]

def debug_backend(gm, sample_inputs):
    captured_gm[0] = gm
    print("=" * 60)
    print("DYNAMO FX GRAPH NODES:")
    print("=" * 60)
    for node in gm.graph.nodes:
        print(f"  op={node.op!r:20s}  target={node.target!r}  name={node.name!r}")
        if node.args:
            print(f"    args = {node.args}")
        if node.kwargs:
            print(f"    kwargs = {node.kwargs}")
    print("=" * 60)
    return gm.forward  # return the original forward

compiled = torch.compile(model, backend=debug_backend)
with torch.no_grad():
    out = compiled(in_0, in_1)

print("Done. Output shapes:", [o.shape for o in out])