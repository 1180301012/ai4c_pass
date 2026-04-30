import torch
from pass_dir.layer_norm_kernel import triton_layer_norm_dispatch


def pattern(in_0, in_1, in_4):
    """
    Match the layer_norm operation for hidden_dim=384:
    tmp_3 = torch.nn.functional.layer_norm(in_4, (384,), in_1, in_0, 1e-12)
    
    Args:
        in_0: bias tensor [384]
        in_1: weight tensor [384]
        in_4: input tensor [batch, seq_len, 384]
    """
    return torch.nn.functional.layer_norm(in_4, (384,), in_1, in_0, 1e-12)


def replacement_args(in_0, in_1, in_4):
    """
    Extract arguments needed for the optimized layer_norm implementation.
    
    Returns:
        Tuple of (bias, weight, input, normalized_shape, route)
    """
    normalized_shape = (in_4.shape[-1],)
    route = "384"
    return (in_0, in_1, in_4, normalized_shape, route)


def replacement_func():
    """Return the shared layer_norm dispatch function."""
    return triton_layer_norm_dispatch