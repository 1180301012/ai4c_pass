import torch

# Pattern matching function - matches the einsum operation
def pattern(in_2, in_1):
    """
    Matches the einsum operation: torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    """
    result = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    return result

# Optimized implementation using PyTorch's built-in matmul
def optimized_einsum_matmul(in_2, in_1):
    """
    More optimized implementation using torch.matmul
    
    einsum('bchw,bchj->bhwj', in_2, in_1) computes:
    For each (b, h, w, j): sum_c in_2[b,h,w,c] * in_1[b,h,c,j]
    
    This is equivalent to: for each (b, h), compute in_2[b,h] @ in_1[b,h].T
    where in_2[b,h] has shape [C, W] and in_1[b,h] has shape [C, J]
    """
    B, C, H, W = in_2.shape
    _, _, _, J = in_1.shape
    
    # Reshape to separate batch and height dimensions
    in_2_reshaped = in_2.view(B * H, C, W)  # [B*H, C, W]
    in_1_reshaped = in_1.view(B * H, C, J)  # [B*H, C, J]
    
    # Compute batched matrix multiplication: [B*H, W, J]
    result = torch.matmul(in_2_reshaped.transpose(1, 2), in_1_reshaped)
    
    # Reshape back to original dimensions: [B, H, W, J]
    result = result.view(B, H, W, J)
    
    return result

# Argument extraction function
def replacement_args(in_2, in_1):
    return (in_2, in_1)

# Replacement function (returns function reference)
def replacement_func():
    return optimized_einsum_matmul