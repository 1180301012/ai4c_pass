import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_fused_norm_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    batch_size,
    seq_len,
    feature_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel with better memory coalescing and reduced shared memory pressure"""
    # Program identifier - each program processes one sequence position
    seq_idx = tl.program_id(0)
    
    # Initialize accumulators for this sequence position (across all features)
    product_sums = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    weight_sums = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process features in blocks for better memory access
    feature_idx_base = 0
    while feature_idx_base < feature_dim:
        # Calculate feature indices for this block
        feature_indices = feature_idx_base + tl.arange(0, BLOCK_SIZE)
        feature_mask = feature_indices < feature_dim
        
        # Process all batch elements for these features
        for batch_idx in range(batch_size):
            # Calculate base offset for this batch and sequence
            base_offset = batch_idx * seq_len * feature_dim + seq_idx * feature_dim
            
            # Load data for all features in this block
            feature_offsets = base_offset + feature_indices
            mask = feature_mask
            
            # Load both tensors (convert in_0 from int64 to float32)
            in_0_vals = tl.load(in_0_ptr + feature_offsets, mask=mask, other=0.0).to(tl.float32)
            in_1_vals = tl.load(in_1_ptr + feature_offsets, mask=mask, other=0.0).to(tl.float32)
            
            # Compute products and weights
            for i in range(BLOCK_SIZE):
                if mask[i]:
                    product_sums[i] += in_0_vals[i] * in_1_vals[i]
                    weight_sums[i] += in_0_vals[i]
        
        feature_idx_base += BLOCK_SIZE
    
    # Apply clamping to avoid division by zero
    for i in range(BLOCK_SIZE):
        if weight_sums[i] != 0.0:
            weight_sums[i] = tl.maximum(weight_sums[i], 1e-09)
    
    # Compute final normalized results
    for i in range(BLOCK_SIZE):
        if weight_sums[i] != 0.0:
            product_sums[i] = product_sums[i] / weight_sums[i]
    
    # Store results - accumulate across all sequences for each feature
    for i in range(BLOCK_SIZE):
        if feature_mask[i]:
            feature_idx = feature_indices[i]
            atomic_op = tl.AtomicAdd
            tl.store(out_ptr + feature_idx, product_sums[i], allow_reordering=False)

@torch.fx.wrap
def optimized_fused_norm_kernel_wrapper(in_0, in_1):
    batch_size, seq_len, feature_dim = in_0.shape
    
    # Create output tensor that will contain the final result [feature_dim]
    out = torch.zeros(feature_dim, dtype=torch.float32, device=in_0.device)
    
    # Configure kernel with optimal block size
    BLOCK_SIZE = 32  # Process features in blocks of 32 for better GPU utilization
    
    # Launch kernel - one program per sequence position (10 programs)
    grid = (seq_len,)
    
    # Execute the optimized fused kernel
    optimized_fused_norm_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        feature_dim=feature_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Match the final operation from original computation:
    # tmp_6 = torch.cat([tmp_5], 1) 
    # Reshape [1024] to [1, 1024] to match original output format
    out = out.unsqueeze(0)
    
    return out

def replacement_func():
    return optimized_fused_norm_kernel_wrapper