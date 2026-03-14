import torch

# Pattern matching for graph 0: (1, 64, -1, 128)
def pattern(input_tensor):
    """
    Match: View -> Transpose pattern
    """
    view_out = input_tensor.view((1, 64, -1, 128))
    transpose_out = view_out.transpose(1, 2)
    return transpose_out

def replacement_args(input_tensor):
    return (input_tensor,)

@torch.fx.wrap
def optimized_view_transpose_1_64(input_tensor):
    """
    Optimized replacement using contiguous memory layout
    """
    B, S, D = input_tensor.shape
    head_dim = 128
    num_heads = D // head_dim
    
    # Reshape and transpose in one go for better memory access
    output = input_tensor.view(B, S, num_heads, head_dim).transpose(1, 2).contiguous()
    return output

def replacement_func():
    return optimized_view_transpose_1_64