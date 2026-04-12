import torch
import triton
import triton.language as tl

# Pattern matching function for RECT_L (uses constant 128)
def pattern(in_0, in_1, in_2):
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
    tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=device(type='cuda'))
    tmp_10 = None
    return (tmp_9, tmp_11)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized fused function using direct operations for RECT_L
@torch.fx.wrap
def optimized_RECT_L_forward(in_0, in_1, in_2):
    """Optimized version for RECT_L case with constant 128"""
    # Direct indexing without intermediates - eliminates tmp_0, tmp_1
    selected_edge_index = in_0[:, in_2]
    
    # Direct concatenation - eliminates tmp_1 intermediate
    concat_result = torch.cat([selected_edge_index, in_1], dim=1)
    
    # Direct ones tensor creation - eliminates tmp_2, tmp_10 intermediates
    mask_count = torch.sum(in_2).item()
    ones_tensor = torch.ones(128 + mask_count, dtype=torch.float32, device='cuda')
    
    return concat_result, ones_tensor

# Replacement function (returns function reference)  
def replacement_func():
    return optimized_RECT_L_forward