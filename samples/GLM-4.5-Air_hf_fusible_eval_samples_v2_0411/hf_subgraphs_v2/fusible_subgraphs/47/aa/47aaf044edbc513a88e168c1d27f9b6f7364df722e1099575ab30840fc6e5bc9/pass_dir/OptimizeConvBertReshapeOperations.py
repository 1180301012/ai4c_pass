import torch

def pattern(x):
    # Two consecutive reshape operations for conv-bert style
    tmp_4 = x.reshape(1, -1, 384, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
    return tmp_5

def replacement_args(x):
    return (x,)

def optimized_convbert_reshape_transform(x):
    """
    Optimized version that combines two reshape operations into one for conv-bert.
    Original pattern: reshape(1, -1, 384, 9) -> reshape(-1, 64, 9)
    This can be simplified: [1, groups, 384, 9] -> [groups*6, 64, 9]
    """
    # Get the original shape after first reshape: [1, groups, 384, 9]
    # Final desired shape: [-1, 64, 9]
    # The transformation is: [1, groups, 384, 9] -> [groups*6, 64, 9]
    
    # Directly reshape to final shape
    # Since 384/64 = 6, we multiply groups by 6
    first_dim = x.shape[1] * 6  # groups * 6
    return x.reshape(first_dim, 64, 9)

def replacement_func():
    # Return the optimized reshape function
    def wrapper(x):
        return optimized_convbert_reshape_transform(x)
    return wrapper