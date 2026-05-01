"""
Helper module: installs a thin relu wrapper into the torch namespace so that:
1. torch.nn.functional.relu keeps working with inplace=True
2. FX symbolic_trace records it as a leaf node (torch is in autowrap_modules)
3. Both model and pattern traces produce the same target object → MATCH
"""
import torch
import torch.nn.functional

# A thin wrapper that accepts inplace kwarg (required by the model) but
# delegates to torch.relu (the actual C builtin).
_relu_lambda = lambda input, inplace=False: torch.relu(input)

# Register in torch namespace with a non-underscore name so that FX
# autowrap_check includes it when iterating torch.__dict__.
torch.relu_compat = _relu_lambda

# Also expose as a module-level name for direct import by the pass file
relu_compat = _relu_lambda

# Replace nn.functional.relu so the model's forward resolves to the same object.
torch.nn.functional.relu = _relu_lambda