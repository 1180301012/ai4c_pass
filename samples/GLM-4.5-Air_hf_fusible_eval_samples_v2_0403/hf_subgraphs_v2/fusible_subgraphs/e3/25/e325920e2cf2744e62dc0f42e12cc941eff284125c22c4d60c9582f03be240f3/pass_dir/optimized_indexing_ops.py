import torch
import triton
import triton.language as tl

# Simple pattern to match index_select operation
def pattern(in_0, in_1):
    tmp_1 = in_0[0]
    tmp_2 = in_1.index_select(-2, tmp_1)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_extract_indices_kernel(
    edge_ptr,
    out_ptr_1,
    out_ptr_0,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for extracting two rows from edge_index tensor"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Extract second row [edge_index[1]]
    offsets_1 = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets_1 < n_elements
    row_1 = tl.load(edge_ptr + 1 * n_elements + offsets_1, mask=mask, other=0)
    
    # Extract first row [edge_index[0]]
    row_0 = tl.load(edge_ptr + 0 * n_elements + offsets_1, mask=mask, other=0)
    
    # Store both rows
    tl.store(out_ptr_1 + offsets_1, row_1, mask=mask)
    tl.store(out_ptr_0 + offsets_1, row_0, mask=mask)

@triton.jit  
def optimized_index_select_kernel(
    features_ptr,
    indices_ptr, 
    out_ptr,
    n_features,
    n_selected,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for indexed feature selection using gather semantics"""
    pid = tl.program_id(0)
    feat_dim = tl.program_id(1)
    
    # Each program handles one feature dimension for a subset of indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_selected
    
    # Load indices for this block
    indices = tl.load(indices_ptr + offsets, mask=mask, other=-1)
    
    # Load features for this feature dimension
    feat_offsets = indices * n_features + feat_dim
    selected_features = tl.load(features_ptr + feat_offsets, mask=(indices >= 0), other=0)
    
    # Store output
    out_offsets = offsets * n_features + feat_dim
    tl.store(out_ptr + out_offsets, selected_features, mask=mask)

@torch.fx.wrap
def optimized_extract_indices(edge_index):
    """Extract both index rows optimized"""
    n_elements = edge_index.shape[1]
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    row_1 = torch.empty(n_elements, dtype=edge_index.dtype, device=edge_index.device)
    row_0 = torch.empty(n_elements, dtype=edge_index.dtype, device=edge_index.device)
    
    optimized_extract_indices_kernel[(num_blocks,)](
        edge_ptr=edge_index,
        out_ptr_1=row_1,
        out_ptr_0=row_0,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return row_1, row_0

@torch.fx.wrap
def optimized_index_select(features, indices):
    """Optimized indexed feature selection"""
    n_selected = indices.shape[0]
    n_features = features.shape[1]
    
    BLOCK_SIZE = 256
    feat_blocks = (n_features + 3) // 4  # Process 4 feature dimensions per program
    num_blocks = (n_selected + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.zeros((n_selected, n_features), dtype=features.dtype, device=features.device)
    
    optimized_index_select_kernel[(num_blocks, feat_blocks)](
        features_ptr=features,
        indices_ptr=indices,
        out_ptr=output,
        n_features=n_features,
        n_selected=n_selected,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@torch.fx.wrap
def optimized_forward(edge_index, features):
    """Combined optimized forward pass"""
    # Extract index rows
    row_1, row_0 = optimized_extract_indices(edge_index)
    
    # Perform optimized index selection
    selected_features = optimized_index_select(features, row_0)
    
    return (row_1, selected_features)

def replacement_func():
    return optimized_forward