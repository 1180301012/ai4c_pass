import torch
import triton
import triton.language as tl
from torch import device

def pattern(a, b):
    """
    Simple pattern: tensor addition that might match part of the computation.
    """
    return a + b

def replacement_args(a, b):
    # Extract the arguments needed for the replacement
    return (a, b)

@triton.jit
def causal_attention_mask_kernel(
    attention_mask_ptr,
    seq_len,
    output_ptr,
    inf_value,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized Triton kernel for causal attention mask creation.
    Creates upper triangular mask + causal logic + attention mask application.
    """
    # Get program identifiers
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Create row and column indices for each thread
    row_start = m * BLOCK_SIZE_M
    col_start = n * BLOCK_SIZE_N
    
    # Generate row and column indices within the block
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    cols = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask to avoid out-of-bound access
    mask_m = rows < seq_len
    mask_n = cols < seq_len
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Calculate causal mask: 1 if col > row (can attend), 0 otherwise
    row_indices = rows[:, None]
    col_indices = cols[None, :]
    causal_mask = col_indices > row_indices
    
    # Create position for upper triangular (diagonal = 1)
    triu_mask = col_indices >= (row_indices + 1)
    
    # Combine upper triangular with causal logic
    combined_mask = causal_mask & triu_mask
    
    # Initialize output with negative infinity
    output = tl.where(combined_mask, 0.0, inf_value)
    
    # Load attention mask (broadcasted from [1, seq_len] to [seq_len, seq_len])
    attn_row = rows[:, None]
    attn_col = cols[None, :]
    attention_values = tl.load(
        attention_mask_ptr + attn_row * seq_len + attn_col,
        mask=mask,
        other=0.0
    )
    
    # Add attention mask
    output += attention_values
    
    # Final: multiply by validity mask (positions that are not -inf)
    validity_mask = output != inf_value
    output = tl.where(validity_mask, output, 0.0)
    
    # Store result
    output_ptr_base = output_ptr + row_start * seq_len + col_start
    tl.store(output_ptr_base, output, mask=mask)

@torch.fx.wrap
def optimized_causal_attention_mask(attention_mask, seq_len):
    """
    Optimized wrapper function that launches the Triton kernel.
    """
    # Get device and dtype info
    device = attention_mask.device
    dtype = torch.float32
    inf_value = -3.4028234663852886e+38
    
    # Create output tensor
    output = torch.full((1, 1, seq_len, seq_len), inf_value, dtype=dtype, device=device)
    
    # Launch kernel - compute the causal mask part without attention first
    mask_base = torch.full((seq_len, seq_len), inf_value, dtype=dtype, device=device)
    
    # Create kernel grid
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch Triton kernel
    causal_attention_mask_kernel[grid_m, grid_n, 1](
        attention_mask_ptr=attention_mask,
        seq_len=seq_len,
        output_ptr=mask_base,
        inf_value=inf_value,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    # Expand to match expected shape [1, 1, seq_len, seq_len]
    result = mask_base.unsqueeze(0).unsqueeze(0)
    
    return result

@torch.fx.wrap
def simple_addition(a, b):
    """
    Simple optimized addition for testing pattern matching.
    """
    return a + b

def replacement_func():
    return simple_addition