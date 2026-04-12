import torch

# Pattern matching for linear output view + transpose fusion
def pattern(linear):
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    return tmp_6

def replacement_args(linear):
    return (linear,)

@torch.fx.wrap
def fuse_view_transpose_linear(linear):
    input_shape = linear.shape
    if len(input_shape) == 3 and input_shape[0] == 1 and input_shape[2] == 512:
        # For this specific pattern: [1, 1, 512] -> [1, 8, 1, 64]
        # Use much simpler, more direct PyTorch operations
        # Instead of view + transpose, use direct reshape + unsqueeze
        
        # Direct approach: [1, 1, 512] -> [1, 8, 64] -> [1, 8, 1, 64]
        # This avoids the intermediate tensor creation that view + transpose requires
        return linear.reshape(1, 8, 64).unsqueeze(2)
    else:
        # Fallback to original operations
        return linear.view(1, 1, -1, 64).transpose(1, 2)

def replacement_func():
    return fuse_view_transpose_linear