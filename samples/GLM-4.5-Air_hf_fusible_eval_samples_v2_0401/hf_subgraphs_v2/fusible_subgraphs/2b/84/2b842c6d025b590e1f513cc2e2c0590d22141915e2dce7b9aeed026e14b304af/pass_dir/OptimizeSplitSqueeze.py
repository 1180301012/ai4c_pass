import torch

def pattern(tmp_2):
    # This matches the split-squeeze-contiguous pattern
    split = tmp_2.split(1, dim = -1)
    tmp_4 = split[0]
    tmp_5 = split[1]
    tmp_6 = tmp_4.squeeze(-1)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_5.squeeze(-1)
    tmp_9 = tmp_8.contiguous()
    return (tmp_7, tmp_9)

def replacement_args(tmp_2):
    return (tmp_2,)

@torch.fx.wrap
def optimized_column_extract(tmp_2):
    # Use direct indexing instead of split for better performance
    # tmp_2 has shape [1, 17, 2]
    # Extract first column: [1, 17, 0:1] then squeeze
    col0 = tmp_2[..., 0:1].squeeze(-1)
    col1 = tmp_2[..., 1:2].squeeze(-1)
    
    # Make contiguous (may not be necessary for direct indexing)
    col0_contiguous = col0.contiguous()
    col1_contiguous = col1.contiguous()
    
    return col0_contiguous, col1_contiguous

def replacement_func():
    return optimized_column_extract