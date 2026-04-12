import torch
import triton
import triton.language as tl

def pattern(in_2):
    # Match the normalization computation pattern
    # tmp_10 = in_2.to(torch.float32)
    tmp_10 = in_2.to(torch.float32)
    # tmp_11 = tmp_10.pow(2)
    tmp_11 = tmp_10.pow(2)
    # tmp_12 = tmp_11.mean(-1, keepdim = True)
    tmp_12 = tmp_11.mean(-1, keepdim = True)
    # tmp_13 = tmp_12 + 1e-06
    tmp_13 = tmp_12 + 1e-06
    # tmp_14 = torch.rsqrt(tmp_13)
    tmp_14 = torch.rsqrt(tmp_13)
    # tmp_15 = tmp_10 * tmp_14
    tmp_15 = tmp_10 * tmp_14
    # Convert to bfloat16
    tmp_16 = tmp_15.to(torch.bfloat16)
    return tmp_16

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def fused_normalization_kernel(
    inputs_ptr,
    weights_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    eps: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Compute program IDs
    batch_idx = tl.program_id(0)
    hidden_idx = tl.program_id(1)
    
    # Create offsets for accessing the input tensor
    # Shape: [batch, seq_len, hidden_dim]
    seq_offset = tl.arange(0, BLOCK_SIZE_X)
    mask_seq = seq_offset < seq_len
    
    # Compute starting pointer for current batch and hidden dimension
    batch_offset = batch_idx * seq_len * hidden_dim
    hidden_offset = hidden_idx * BLOCK_SIZE_X
    base_offset = batch_offset + hidden_offset
    
    # Power sequence: load input, square, compute mean along sequence dimension
    for i in range(0, seq_len, BLOCK_SIZE_X):
        seq_idx = i + seq_offset
        seq_mask = seq_idx < seq_len
        
        # Load input values for current position
        input_ptr = inputs_ptr + base_offset + seq_idx
        inputs = tl.load(input_ptr, mask=seq_mask, other=0.0)
        
        # Square the inputs and accumulate for mean computation
        inputs_sq = inputs * inputs
        
        # For the first iteration, initialize sum, otherwise accumulate
        if i == 0:
            sum_sq = inputs_sq
        else:
            sum_sq += inputs_sq
    
    # Compute mean along sequence dimension
    mean_sq = sum_sq / seq_len
    
    # Add epsilon and compute rsqrt
    mean_sq_eps = mean_sq + eps
    inv_std = tl.rsqrt(mean_sq_eps)
    
    # Apply normalization: multiply by inverse standard deviation
    normalized = inputs * inv_std
    
    # Store results
    output_ptr_base = output_ptr + base_offset
    for i in range(0, seq_len, BLOCK_SIZE_X):
        seq_idx = i + seq_offset
        seq_mask = seq_idx < seq_len
        output_offset = output_ptr_base + seq_idx
        tl.store(output_offset, seq_mask, normalized)

@torch.fx.wrap  
def fused_normalization(inputs_embeds, in_0, eps=1e-06):
    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    
    # Reshape weights to match for broadcast
    if len(in_0.shape) == 1:
        # in_0 is [hidden_dim], reshape to [1, 1, hidden_dim] for broadcasting
        weights = in_0.view(1, 1, hidden_dim)
    else:
        weights = in_0
        
    # Create output tensor
    output = torch.empty_like(inputs_embeds, dtype=torch.bfloat16)
    
    # Determine block sizes for optimal GPU utilization  
    BLOCK_SIZE_X = min(256, seq_len)  # Block size along sequence dimension
    BLOCK_SIZE_Y = min(256, hidden_dim)  # Block size along hidden dimension
    
    # Calculate grid dimensions
    num_batches = batch_size
    num_hidden_blocks = (hidden_dim + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid = (num_batches, num_hidden_blocks)
    
    # Launch Triton kernel
    fused_normalization_kernel[grid](
        inputs_ptr=inputs_embeds,
        weights_ptr=weights.float(),  # Convert to float32 for computation
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        eps=eps,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return output

def replacement_func():
    # Create a partial function that includes the optimization
    def optimized_wrapper(inputs_embeds):
        # Extract normalization weights from the model (this will be in_0 in the actual graph)
        # Note: This is a simplified version - in real implementation, weights would be passed differently
        return fused_normalization(inputs_embeds, torch.ones(2048, device=inputs_embeds.device), eps=1e-06)
    
    return optimized_wrapper