import torch
import triton
import triton.language as tl

# ============================================================
# Pattern: Single identity dropout (p=0, training=False)
# ============================================================
def pattern(x):
    return torch.nn.functional.dropout(x, 0.0, False, False)

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def identity_passthrough_single(x):
    return x

def replacement_func():
    return identity_passthrough_single