import torch
from pass_dir.shared_kernels import fused_dispatch_wrapper, ROUTE_MULT_ADD_LN


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern matching the computation:
    1. embedding(in_4, in_1) * 16.0 -> tmp_5
    2. arange(0, 1).expand(1, -1) + 2 -> tmp_8
    3. embedding(tmp_8, in_0) -> tmp_9
    4. tmp_5 + tmp_9 -> tmp_10
    5. layer_norm(tmp_10) -> tmp_11
    6. dropout(tmp_11) -> tmp_12
    """
    from torch import device
    
    # First embedding branch
    tmp_4 = torch.nn.functional.embedding(in_4, in_1, 1, None, 2.0, False, False)
    tmp_5 = tmp_4 * 16.0
    
    # Position sequence generation
    tmp_6 = torch.arange(0, 1, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_7 = tmp_6.expand(1, -1)
    tmp_8 = tmp_7 + 2
    
    # Second embedding branch
    tmp_9 = torch.nn.functional.embedding(tmp_8, in_0, None, None, 2.0, False, False)
    
    # Combine and normalize
    tmp_10 = tmp_5 + tmp_9
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_3, in_2, 1e-05)
    
    # Dropout (training=False, so essentially identity)
    tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.1, training=False)
    
    return tmp_12


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments for the fused kernel.
    """
    return (in_0, in_1, in_2, in_3, in_4, ROUTE_MULT_ADD_LN)


def replacement_func():
    """
    Returns the shared dispatch wrapper function.
    """
    return fused_dispatch_wrapper