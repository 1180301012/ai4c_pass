import torch

def pattern(x, a, b, c):
    """Match ReLU -> Reshape -> Permute pattern with explicit arguments"""
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    tmp_2 = tmp_0.reshape(a, b, c)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_3

def replacement_args(x, target_shape):
    """Extract arguments for the fusion"""
    # Only fuse for specific pattern: 4D input, 3D reshape, last dim is 1
    original_shape = x.shape
    if (len(original_shape) != 4 or original_shape[-1] != 1 or 
        len(target_shape) != 3):
        # Not the pattern we want to fuse, fallback
        return (x, target_shape)
    
    # Return individual arguments for the pattern
    return (x, target_shape[0], target_shape[1], target_shape[2])

def fused_optimization(x, a, b, c):
    """Fused ReLU + Squeeze + Reshape + Permute"""
    # When input is [..., 1] and we're reshaping to [a, b, c] then permuting to [a, c, b]
    # We can fuse all operations
    return torch.relu(x).squeeze(-1).reshape(a, b, c).permute(0, 2, 1)

def replacement_func():
    """Return fused function"""
    return fused_optimization