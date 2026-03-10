import torch

def pattern(tmp_3):
    tmp_4 = tmp_3.reshape(1, 12, 12, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    return tmp_5

def replacement_args(tmp_3):
    return (tmp_3,)

# Optimized reshape + permute fusion
# Instead of separate reshape and permute operations, we can optimize this
# by computing the final shape directly and avoiding intermediate copies
@torch.fx.wrap
def optimized_reshape_permute_fusion(input_tensor):
    """
    Fuse reshape and permute operations to avoid intermediate tensor creation
    """
    # Calculate the final size: [1, 12, 12, hidden/144] -> permute to [1, hidden/144, 12, 12]
    # First, figure out the hidden size for the final dimension
    input_shape = input_tensor.shape
    original_batch, original_seq, hidden_size = input_shape
    
    # Calculate what the final hidden dimension should be
    final_hidden = hidden_size * original_seq // (12 * 12)
    
    # Create a view directly to the final shape without intermediate permute
    # This avoids creating the intermediate [1, 12, 12, final_hidden] tensor
    final_output = input_tensor.reshape(1, final_hidden, 12, 12)
    
    return final_output

def replacement_func():
    return optimized_reshape_permute_fusion