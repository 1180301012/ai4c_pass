import torch
from pass_dir.shared_kernel import optimized_matmul


def pattern(in_0, in_1, in_2):
    """
    Match pattern for reshape to [-1, 16]:
    1. matmul(in_1, in_0) - [B, M, K] @ [B, K, 1] -> [B, M]
    2. reshape(matmul, [-1, 16])
    3. transpose(in_2, -1, -2)
    
    Returns (reshape_result, transpose_result)
    """
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 16])
    tmp_2 = in_2.transpose(-1, -2)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement."""
    return (in_0, in_1, in_2, 16)


def wrapper_16(in_0, in_1, in_2, reshape_dim):
    """
    Fused wrapper: matmul + reshape + transpose
    
    Computes:
    - matmul(in_1, in_0) -> reshape to [-1, reshape_dim]
    - transpose(in_2, -1, -2)
    """
    # Use optimized matmul
    matmul_result = optimized_matmul(in_1, in_0)
    
    # Compute reshape output shape
    B, M = matmul_result.shape
    num_elements = B * M
    new_first_dim = num_elements // reshape_dim
    reshape_result = matmul_result.view(new_first_dim, reshape_dim)
    
    # Transpose remains as-is (cheap operation)
    transpose_result = in_2.transpose(-1, -2)
    
    return (reshape_result, transpose_result)


def replacement_func():
    return wrapper_16