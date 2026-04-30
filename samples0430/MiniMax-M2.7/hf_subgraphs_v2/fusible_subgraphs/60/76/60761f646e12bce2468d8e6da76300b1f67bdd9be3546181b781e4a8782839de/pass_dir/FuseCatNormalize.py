import torch
import triton
import triton.language as tl

@triton.jit
def fused_cat_normalize_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr = 1e-12,
):
    """
    Fused kernel for:
    1. tmp_0 = torch.cat([in_0], 1)  # Concatenate with itself along dim=1
    2. tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    
    Optimized approach: Instead of materializing the cat result, we directly compute:
    - norm = sqrt(2) * sqrt(sum(input^2)) = sqrt(2 * sum(input^2))
    - output[i] = input[i] / norm  (for both halves)
    
    This avoids:
    - Memory allocation for intermediate concatenated tensor
    - Memory copy for cat operation
    - Separate normalize kernel launch
    """
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Compute stride for accessing batch elements
    input_offset = batch_idx * feature_dim
    
    # First pass: Compute sum of squares for L2 norm
    # We need sqrt(2) * sqrt(sum(x^2)), so we compute sum(x^2) first
    sum_squares = 0.0
    for i in range(0, feature_dim, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < feature_dim
        
        # Load input values
        x = tl.load(input_ptr + input_offset + offs, mask=mask, other=0.0)
        
        # Accumulate sum of squares
        sum_squares += tl.sum(x * x)
    
    # Compute the norm: sqrt(2) * sqrt(sum_squares)
    # Avoid overflow by dividing before multiplying
    if feature_dim > 0:
        norm = tl.sqrt(sum_squares * 2.0 + EPS * 2.0)
    else:
        norm = 1.0
    
    # Second pass: Normalize and write output (both halves)
    # Output is [input, input] / norm along dim=1
    # So output[0:feature_dim] = input / norm
    # And output[feature_dim:2*feature_dim] = input / norm
    
    output_base = batch_idx * (feature_dim * 2)
    
    for i in range(0, feature_dim, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < feature_dim
        
        # Load input values
        x = tl.load(input_ptr + input_offset + offs, mask=mask, other=0.0)
        
        # Normalize
        normalized = x / norm
        
        # Write first half: output[0:feature_dim] = normalized
        tl.store(output_ptr + output_base + offs, normalized, mask=mask)
        
        # Write second half: output[feature_dim:2*feature_dim] = normalized
        tl.store(output_ptr + output_base + feature_dim + offs, normalized, mask=mask)


@torch.fx.wrap
def fused_cat_normalize_wrapper(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the fused cat + normalize kernel.
    Takes input [batch, feature_dim] and returns [batch, feature_dim*2] normalized.
    """
    batch_size, feature_dim = input_tensor.shape
    output_shape = (batch_size, feature_dim * 2)
    
    # Allocate output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Define block size - optimized for typical BERT hidden size (768)
    BLOCK_SIZE = 1024
    
    # Launch kernel: one program per batch element
    num_programs = batch_size
    
    # Specialize for common feature dimensions
    if feature_dim == 768:
        fused_cat_normalize_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output,
            batch_size=batch_size,
            feature_dim=feature_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Fallback for other dimensions
        fused_cat_normalize_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output,
            batch_size=batch_size,
            feature_dim=feature_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output


def pattern(in_0):
    """
    Pattern: cat + normalize fusion
    Matches: 
        tmp_0 = torch.cat([in_0], 1)
        tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    Returns both intermediate and final result (required for subgraph matching)
    """
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return tmp_0, tmp_1


def replacement_args(in_0):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_0,)


def replacement_func():
    """
    Returns the replacement function.
    """
    return fused_cat_normalize_wrapper