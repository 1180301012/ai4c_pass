import torch

def pattern(tmp_12):
    # 96-channel case: [1, 32, 8, 32, 8, 96] -> [1, 32, 32, 8, 8, 96]  
    tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5)
    return tmp_13

def replacement_args(tmp_12):
    return (tmp_12,)

@torch.fx.wrap
def optimized_permute_96(input_tensor):
    # Apply the optimized permute operation
    # For Swin Transformer, this permutes [B, H, W//window_size, window_size//patch_size, window_size//patch_size, C]
    # to [B, H, H//window_size, window_size, window_size, C] for window attention
    output = input_tensor.permute(0, 1, 3, 2, 4, 5)
    
    return output

def replacement_func():
    return optimized_permute_96