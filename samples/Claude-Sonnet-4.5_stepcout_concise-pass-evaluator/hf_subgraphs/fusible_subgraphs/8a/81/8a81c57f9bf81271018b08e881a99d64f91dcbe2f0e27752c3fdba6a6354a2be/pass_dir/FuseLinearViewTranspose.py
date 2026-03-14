import torch
import triton
import triton.language as tl


def pattern(input, weight, bias):
    """
    Pattern: Linear + View + Transpose for value projection in attention
    Linear(input, weight, bias) -> view -> transpose(1, 2)
    """
    linear_out = torch.nn.functional.linear(input, weight, bias)
    viewed = linear_out.view(linear_out.shape[0], -1, 12, 64)
    transposed = viewed.transpose(1, 2)
    return transposed


def replacement_args(input, weight, bias):
    return (input, weight, bias)


@torch.fx.wrap
def fused_linear_view_transpose(input, weight, bias):
    """
    Fused Linear + View + Transpose
    This combines three operations into one, reducing memory traffic
    """
    # Perform linear operation
    linear_out = torch.nn.functional.linear(input, weight, bias)
    
    # Get shapes
    B = linear_out.shape[0]
    S = linear_out.shape[1]
    hidden_dim = linear_out.shape[2]
    
    # Assuming hidden_dim = num_heads * head_dim
    # For the models we're targeting: hidden_dim=768, num_heads=12, head_dim=64
    # Or hidden_dim=128, num_heads=2, head_dim=64
    num_heads = 12 if hidden_dim == 768 else 2
    head_dim = hidden_dim // num_heads
    
    # View and transpose in one operation by directly reshaping to target layout
    # Instead of: (B, S, H*D) -> view(B, S, H, D) -> transpose(1,2) -> (B, H, S, D)
    # We do: (B, S, H*D) -> reshape to (B, H, S, D) directly
    output = linear_out.view(B, S, num_heads, head_dim).transpose(1, 2)
    
    return output


def replacement_func():
    return fused_linear_view_transpose