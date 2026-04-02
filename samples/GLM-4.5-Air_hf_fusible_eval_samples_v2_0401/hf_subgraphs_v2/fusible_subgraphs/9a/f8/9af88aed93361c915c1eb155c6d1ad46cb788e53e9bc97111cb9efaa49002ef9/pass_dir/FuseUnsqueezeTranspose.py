import torch

def pattern(in_0):
    tmp_1 = in_0.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

@torch.fx.wrap
def direct_reshape_transpose(in_0):
    """
    Directly reshape and transpose to achieve the final result.
    Input: [batch, channels, features] 
    After unsqueeze(1): [batch, 1, channels, features]
    After transpose(2, 3): [batch, 1, features, channels]
    
    This can be done directly as: reshape and then swap last two dims
    """
    # Same as: return in_0.unsqueeze(1).transpose(2, 3)
    # But we'll implement it as a single operation
    batch_size = in_0.shape[0]
    channels = in_0.shape[1]
    features = in_0.shape[2]
    
    # Create output with the final transposed shape
    # This is equivalent to: [batch, 1, features, channels]
    return in_0.reshape(batch_size, 1, channels, features).transpose(-1, -2)

def replacement_func():
    return direct_reshape_transpose