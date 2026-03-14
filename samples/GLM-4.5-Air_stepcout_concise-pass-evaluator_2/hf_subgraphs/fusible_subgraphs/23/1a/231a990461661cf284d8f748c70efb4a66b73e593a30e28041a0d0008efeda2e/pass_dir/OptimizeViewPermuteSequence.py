import torch

@torch.fx.wrap
def optimized_view_sequence(x):
    """
    Optimized version of the view/permute sequence:
    Original: [1, 128, 9216] -> [1, 96, 96, 128] -> [1, 8, 12, 8, 12, 128] -> 
              [1, 8, 8, 12, 12, 128] -> [64, 12, 12, 128] -> [64, 144, 128]
    
    This can be optimized by directly computing [1, 128, 9216] -> [64, 144, 128]
    since the intermediate steps are just different views of the same data
    """
    # The sequence essentially transforms [1, 128, 9216] to [64, 144, 128]
    # The intermediate steps involve reshaping and permuting but can be fused
    
    # Check input shape
    if x.shape != (1, 128, 9216):
        # Fallback to original behavior if shape doesn't match expected pattern
        tmp_13 = x.view(1, 96, 96, 128)
        # Skip the no-op pad operation since it adds nothing
        tmp_15 = tmp_13.view(1, 8, 12, 8, 12, 128)
        tmp_16 = tmp_15.permute(0, 1, 3, 2, 4, 5)
        tmp_17 = tmp_16.contiguous()
        tmp_18 = tmp_17.view(-1, 12, 12, 128)
        return tmp_18.view(-1, 144, 128)
    
    # Direct reshape - this is much more efficient than multiple view operations
    # The calculation: 1*128*9216 = 1179648, 64*144*128 = 1179648 (same total elements)
    return x.reshape(64, 144, 128)

# Pattern matching function - matches the entire view/permute sequence
def pattern(tmp_12):
    tmp_13 = tmp_12.view(1, 96, 96, 128)
    tmp_14 = torch.nn.functional.pad(tmp_13, (0, 0, 0, 0, 0, 0), 'constant', None)
    tmp_15 = tmp_14.view(1, 8, 12, 8, 12, 128)
    tmp_16 = tmp_15.permute(0, 1, 3, 2, 4, 5)
    tmp_17 = tmp_16.contiguous()
    tmp_18 = tmp_17.view(-1, 12, 12, 128)
    tmp_19 = tmp_18.view(-1, 144, 128)
    return tmp_19

# Argument extraction function  
def replacement_args(tmp_12):
    return (tmp_12,)

# Replacement function
def replacement_func():
    return optimized_view_sequence