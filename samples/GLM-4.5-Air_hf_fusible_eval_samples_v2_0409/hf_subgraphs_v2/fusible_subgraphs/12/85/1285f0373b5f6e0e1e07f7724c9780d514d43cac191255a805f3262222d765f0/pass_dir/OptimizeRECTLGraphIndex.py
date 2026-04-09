import torch

def pattern(in_0, in_1, in_2):
    """Match RECT_L graph computation pattern with constant 128"""
    tmp_0 = in_0
    tmp_1 = tmp_0[slice(None, None, None), in_2]
    tmp_0 = None
    tmp_2 = torch.ops.aten.sym_size.int(tmp_1, 1)
    tmp_3 = torch._check_is_size(tmp_2)
    tmp_3 = None
    tmp_4 = tmp_2 >= 0
    tmp_5 = torch.ops.aten._assert_scalar.default(tmp_4, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'")
    tmp_4 = tmp_5 = None
    tmp_6 = tmp_2 <= 128
    tmp_7 = torch.ops.aten._assert_scalar.default(tmp_6, "Runtime assertion failed for expression u0 <= 128 on node 'le_1'")
    tmp_6 = tmp_7 = None
    tmp_8 = torch._check_is_size(tmp_2)
    tmp_8 = None
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    tmp_1 = None
    tmp_10 = torch.sym_sum([128, tmp_2])
    tmp_2 = None
    tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=torch.device('cuda'))
    tmp_10 = None
    return (tmp_9, tmp_11)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the optimized kernel"""
    return (in_0, in_1, in_2)

@torch.fx.wrap
def optimized_rectl_fusion(in_0, in_1, in_2):
    """Optimized fusion for RECT_L graph with constant 128"""
    # Step 1: Apply boolean indexing to select relevant edges - more efficient than original
    selected_edges = in_0[:, in_2]  # Shape: [2, K] where K = number of True values in mask
    
    # Step 2: Concatenate with loop indices
    concatenated = torch.cat([selected_edges, in_1], dim=1)  # Shape: [2, K + M]
    
    # Step 3: Calculate final size and create ones tensor - fused operation
    k = selected_edges.shape[1]
    final_size = 128 + k  # Hardcoded constant for RECT_L
    ones_tensor = torch.ones(final_size, dtype=torch.float32, device=torch.device('cuda'))
    
    return concatenated, ones_tensor

def replacement_func():
    """Return the optimized function reference for RECT_L"""
    return optimized_rectl_fusion