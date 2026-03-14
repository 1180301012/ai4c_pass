import torch

def pattern(in_0, in_1):
    """Pattern: exact match for graph 1 with split [1, 196] and view [1, 384, 14, 14]"""
    # Element-wise addition
    tmp_0 = in_1 + in_0
    
    # Split with exact parameters for graph 1
    tmp_1 = torch.functional.split(tmp_0, [1, 196], 1)
    
    # Get both parts
    tmp_2 = tmp_1[0]
    tmp_3 = tmp_1[1]
    
    # Permute last two dimensions
    tmp_4 = tmp_3.permute(0, 2, 1)
    
    # View with exact parameters for graph 1
    tmp_5 = tmp_4.view(1, 384, 14, 14)
    
    # Return both observable outputs
    return tmp_2, tmp_5

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def optimize_graph1_wrapper(in_0, in_1):
    """Optimized implementation for graph 1 using efficient operations"""
    # Efficient fused computation for graph 1
    added = in_0 + in_1
    
    # Extract both parts efficiently without split overhead
    output_0 = added[:, :1, :].contiguous()
    output_1 = added[:, 1:, :].transpose(1, 2).contiguous()
    output_1 = output_1.reshape(1, 384, 14, 14)
    
    return output_0, output_1

def replacement_func():
    return optimize_graph1_wrapper