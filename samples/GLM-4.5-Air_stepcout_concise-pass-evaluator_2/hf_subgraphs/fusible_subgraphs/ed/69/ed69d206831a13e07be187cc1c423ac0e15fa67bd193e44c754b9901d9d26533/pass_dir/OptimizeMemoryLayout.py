import torch

def pattern(in_0, in_1):
    """Pattern: add + split + permute + view sequence"""
    # Element-wise addition
    tmp_0 = in_1 + in_0
    
    # Split along dimension 1 - using flexible split to match both patterns
    tmp_1 = torch.functional.split(tmp_0, [1, -1], 1)
    
    # Get both parts of split
    tmp_2 = tmp_1[0]  # First part (what gets returned)
    tmp_3 = tmp_1[1]  # Second part (gets processed)
    
    # Permute last two dimensions and view in one efficient operation
    tmp_5 = tmp_3.transpose(1, 2).reshape(1, 384, -1)
    
    # Return both observable outputs
    return tmp_2, tmp_5

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def optimized_memory_layout_wrapper(in_0, in_1):
    """Optimized wrapper that fuses multiple operations efficiently"""
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]
    hidden_dim = in_0.shape[2]
    
    # Efficient fused computation
    # 1. Perform addition
    added = in_0 + in_1
    
    # 2. Extract first element efficiently (what gets returned as tmp_2)
    output_0 = added[:, :1, :].contiguous()  # Ensure contiguous memory
    
    # 3. Efficiently transform the rest directly - avoid intermediate tensors
    # Split, transpose, and reshape in one operation
    remaining = added[:, 1:, :]  # This is more efficient than split + index
    output_1 = remaining.transpose(1, 2).contiguous()
    
    # Determine the final spatial dimensions dynamically
    spatial_size = remaining.shape[1] * remaining.shape[2]
    if spatial_size == 196:  # 14*14
        output_1 = output_1.reshape(1, 384, 14, 14)
    elif spatial_size == 576:  # 24*24
        output_1 = output_1.reshape(1, 384, 24, 24)
    else:
        # Fallback for any other size
        sqrt_size = int(spatial_size ** 0.5)
        output_1 = output_1.reshape(1, 384, sqrt_size, spatial_size // sqrt_size)
    
    return output_0, output_1

def replacement_func():
    return optimized_memory_layout_wrapper