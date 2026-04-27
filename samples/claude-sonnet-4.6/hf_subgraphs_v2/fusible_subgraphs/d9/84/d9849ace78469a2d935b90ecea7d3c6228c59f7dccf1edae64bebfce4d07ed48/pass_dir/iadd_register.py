"""
Helper module: registers operator.iadd as a torch.fx leaf call_function.
Importing this module is a side-effect that ensures the torch.fx pattern
tracer records  call_function(operator.iadd, ...)  instead of tracing
through operator.iadd into the underlying __iadd__ / add calls.
"""
import operator
import torch

# Must be at true module top level (not inside a function) per torch.fx rules.
torch.fx.wrap(operator.iadd)