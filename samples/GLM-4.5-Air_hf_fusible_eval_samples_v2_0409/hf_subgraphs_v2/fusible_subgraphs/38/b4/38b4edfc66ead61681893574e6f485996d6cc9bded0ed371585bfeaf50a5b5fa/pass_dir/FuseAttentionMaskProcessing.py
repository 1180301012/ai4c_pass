import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function for attention mask processing
def pattern(triangular_mask, input_mask):
    """
    Match the attention mask processing pattern:
    tmp_7 = triangular_mask.to(torch.float32)
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, n, n)
    tmp_10 = input_mask[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, n, n)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_13 = torch.tensor(1.0, dtype=torch.float32)
    tmp_14 = tmp_13 - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    """
    n = triangular_mask.shape[0]  # Get size from triangular mask
    
    # Convert triangular mask to float32 and expand
    tmp_7 = triangular_mask.to(torch.float32)
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, n, n)
    
    # Process input mask
    tmp_10 = input_mask[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, n, n)
    tmp_12 = tmp_11.to(torch.float32)
    
    # Create combined mask
    tmp_13 = torch.tensor(1.0, dtype=torch.float32)
    tmp_14 = tmp_13 - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    
    # Apply final mask
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    
    return tmp_19

# Argument extraction function
def replacement_args(*args):
    """
    Extract arguments needed for the optimized kernel:
    - triangular_mask: the pre-computed triangular mask
    - input_mask: the input attention mask
    """
    # Find triangular_mask (2D tensor)
    triangular_mask = None
    input_mask = None
    
    for tensor in args:
        if tensor is not None:
            if len(tensor.shape) == 2 and hasattr(tensor, 'dtype'):
                if triangular_mask is None:
                    triangular_mask = tensor
                elif input_mask is None and len(input_mask.shape) == 1:
                    input_mask = tensor
            elif len(tensor.shape) == 1:
                input_mask = tensor
    
    return (triangular_mask, input_mask)

@triton.jit
def process_attention_mask_kernel(
    triangular_mask_ptr,
    input_mask_ptr,
    out_ptr,
    n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel to process attention masks in a single fused operation
    """
    # Get program IDs
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Create offsets within the block
    offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create 2D grid of positions
    row_mask = offs_m[:, None] < n
    col_mask = offs_n[None, :] < n
    mask = row_mask & col_mask
    
    # Load triangular mask values
    triangular_offsets = offs_m[:, None] * n + offs_n[None, :]
    triangular_values = tl.load(
        triangular_mask_ptr + triangular_offsets,
        mask=mask,
        other=0.0
    )
    
    # Load and process input mask
    input_offsets = offs_n  # Input mask is 1D
    input_values = tl.load(
        input_mask_ptr + input_offsets,
        mask=offs_n < n,
        other=0
    ).to(tl.float32)
    
    # Create combined mask: 1.0 - input_mask, where 0 indicates valid positions
    combined_mask = 1.0 - input_values
    
    # Convert to boolean (True where should be masked)
    should_mask = combined_mask == 1.0
    
    # Apply mask to triangular values: fill with -inf where should_mask is True
    result_values = tl.where(should_mask[:, None], -3.4028234663852886e+38, triangular_values)
    
    # Store result
    tl.store(
        out_ptr + triangular_offsets,
        result_values,
        mask=mask
    )

@torch.fx.wrap
def process_attention_mask_optimized(triangular_mask, input_mask):
    """
    Optimized function to process attention masks
    """
    n = triangular_mask.shape[0]
    
    # Create output tensor [1, 1, n, n] but flatten for kernel access
    out = torch.empty((n, n), dtype=torch.float32, device='cuda')
    
    # Calculate grid dimensions
    grid_m = (n + 63) // 64  # Use 64x64 blocks for better GPU utilization
    grid_n = (n + 63) // 64
    
    # Launch the optimized kernel
    process_attention_mask_kernel[(grid_m, grid_n)](
        triangular_mask_ptr=triangular_mask,
        input_mask_ptr=input_mask,
        out_ptr=out,
        n=n,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
    )
    
    # Reshape to final format [1, 1, n, n]
    return out.unsqueeze(0).unsqueeze(0)

# Replacement function (returns function reference, not a call)
def replacement_func():
    return process_attention_mask_optimized