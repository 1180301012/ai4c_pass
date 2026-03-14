import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, device_info):
    """
    Exact pattern matching the original computation from model.py:
    tmp_0 = in_0
    tmp_1 = torch.arange(128, dtype=torch.int64, device=device_info)  # Use concrete value for pattern matching
    tmp_2 = tmp_1.view(1, -1)
    tmp_3 = in_1 - tmp_2
    tmp_4 = tmp_3 + 2048
    tmp_5 = tmp_4 - 1
    tmp_6 = torch.nn.functional.embedding(tmp_5, in_0, None, None, 2.0, False, False)
    tmp_7 = tmp_6.to(dtype=torch.float32)
    return tmp_7
    """
    tmp_0 = in_0
    tmp_1 = torch.arange(128, dtype=torch.int64, device=device_info)
    tmp_2 = tmp_1.view(1, -1)
    tmp_3 = in_1 - tmp_2
    tmp_4 = tmp_3 + 2048
    tmp_5 = tmp_4 - 1
    tmp_6 = torch.nn.functional.embedding(tmp_5, tmp_0, None, None, 2.0, False, False)
    tmp_7 = tmp_6.to(dtype=torch.float32)
    return tmp_7

def replacement_args(in_0, in_1, device_info):
    return (in_0, in_1, device_info)

@triton.autotune(
    configs=[
        triton.Config(num_warps=1, num_stages=1),
        triton.Config(num_warps=2, num_stages=1), 
        triton.Config(num_warps=2, num_stages=2),
        triton.Config(num_warps=4, num_stages=2),
        triton.Config(num_warps=4, num_stages=3),
        triton.Config(num_warps=8, num_stages=3),
    ],
    key=['position_ids_n', 'position_ids_m'],
)
@torch.heuristics({
    "BLOCK_SIZE": lambda args: min(256, args["position_ids_m"])
})
@triton.jit
def optimized_embedding_kernel(
    position_ids_ptr, 
    weight_ptr,
    output_ptr,
    position_ids_n,  # number of rows in position_ids  
    position_ids_m,  # number of columns in position_ids
    weight_features, # number of features (64)
    num_embeddings,  # number of embeddings (4095)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of position_ids
    row_idx = tl.program_id(0)
    col_mask = tl.arange(0, BLOCK_SIZE) < position_ids_m
    
    # Load position_ids for this row
    offset = row_idx * position_ids_m
    position_ids = tl.load(position_ids_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=col_mask, other=0).to(tl.int64)
    
    # Pre-compute indices: position_ids + 2047 (consolidates - arange + 2048 - 1)
    indices = position_ids + 2047
    
    # Ensure indices are within bounds
    indices = tl.where((indices >= 0) & (indices < num_embeddings), indices, 0)
    
    # Load embedding weights for each feature dimension
    embedding_vectors = tl.load(
        weight_ptr + indices[:, None] * weight_features + tl.arange(0, weight_features)[None, :],
        mask=col_mask[:, None] & (indices[:, None] >= 0),
        other=0.0
    )
    
    # The embedding values are already float32, so no conversion needed
    output_vectors = embedding_vectors
    
    # Store the final result
    tl.store(output_ptr + offset + tl.arange(0, BLOCK_SIZE)[:, None], output_vectors, mask=col_mask[:, None])

@torch.fx.wrap
def optimized_forward_embedding(in_0, in_1, device_info):
    """
    Optimized forward pass that fuses arithmetic operations directly into embedding lookup
    Original: in_1 - arange + 2048 - 1 = in_1 + 2047
    """
    position_ids = in_1  # Shape: [N, 1] (128,1) or (512,1)
    weight = in_0       # Shape: [4095, 64]
    
    N, M = position_ids.shape
    num_embeddings, weight_features = weight.shape
    
    # Output shape: [N, weight_features] 
    output = torch.empty((N, weight_features), dtype=torch.float32, device=device_info)
    
    # Launch kernel - handle both 128 and 512 cases
    optimized_embedding_kernel[(N, 1)](
        position_ids_ptr=position_ids,
        weight_ptr=weight,
        output_ptr=output,
        position_ids_n=N,
        position_ids_m=M,
        weight_features=weight_features,
        num_embeddings=num_embeddings,
        BLOCK_SIZE=M  # Use M since we're processing entire row at once
    )
    
    return output

def replacement_func():
    """Returns the optimized function that fuses arithmetic into embedding lookup"""
    return optimized_forward_embedding