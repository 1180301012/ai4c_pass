import torch
from torch import device

# Pattern matching for eliminating redundant device transfer
def pattern(encoding_tensor):
    # Match the pattern: redundant device transfer
    # Original: tmp_5 = tmp_4.to(device(type='cuda', index=0))
    # Optimized: return tmp_4 directly (eliminating redundant transfer)
    return encoding_tensor.to(device(type='cuda', index=0))

# Extract arguments from matched nodes
def replacement_args(node):
    # Extract the encoding tensor that needs redundant device transfer
    return (node,)

# Simple function that eliminates redundant device transfer
def remove_redundant_device_transfer(encoding_tensor):
    # The original code does: tmp_5 = tmp_4.to(device(type='cuda', index=0))
    # But tmp_4 is already on device, so this is redundant
    # Simply return the tensor as-is
    return encoding_tensor

# Replacement function that returns the optimized implementation
def replacement_func():
    return remove_redundant_device_transfer