import torch

# Import shared implementation
from pass_dir._shared_impl import fused_gelu_trans_add_layernorm_dispatch

# Pattern matching function - matches float16/float32 graphs with dropout p=0.1
# Matches from conv1d output through slice, gelu, transpose, add, dropout (identity), layer_norm
def pattern(conv_output, hidden_states, ln_weight, ln_bias):
    sliced = conv_output[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))]
    gelu_out = torch.nn.functional.gelu(sliced)
    transposed = gelu_out.transpose(1, 2)
    added = hidden_states + transposed
    dropped = torch.nn.functional.dropout(added, 0.1, False, False)
    normalized = torch.nn.functional.layer_norm(dropped, (1024,), ln_weight, ln_bias, 1e-05)
    return (dropped, normalized)

# Argument extraction function - includes route string for dispatch
def replacement_args(conv_output, hidden_states, ln_weight, ln_bias):
    return (conv_output, hidden_states, ln_weight, ln_bias, "p010")

# Replacement function - returns the shared dispatch wrapper (same object across all passes)
def replacement_func():
    return fused_gelu_trans_add_layernorm_dispatch