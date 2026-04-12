import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching for the type conversion and expansion section
def pattern(in_1, in_3):
    """
    Pattern matching the type conversion and expansion section:
    Multiple redundant float conversions, unnecessary device transfers, multiple expand operations.
    """
    # First tensor processing chain with redundant operations
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_15 = None
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_16 = None
    tmp_18 = tmp_17.to(device=device(type='cuda', index=0))  # Redundant device transfer
    tmp_17 = None
    tmp_21 = tmp_18.float()  # Redundant second float conversion
    tmp_18 = None

    # Second tensor processing chain with redundant operations  
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_19 = None
    tmp_22 = tmp_20.float()  # Redundant second float conversion
    tmp_20 = None

    return tmp_21, tmp_22

# Argument extraction for the replacement
def replacement_args(in_1, in_3):
    """
    Extract arguments needed for optimization
    """
    return (in_1, in_3)

# Optimized kernel implementation
@triton.jit
def type_conversion_kernel(
    tensor1_ptr,           # Input tensor [hidden_dim]
    tensor2_ptr,           # Input tensor [batch_size, seq_len] 
    output1_ptr,           # Output [1, hidden_dim, 1] 
    output2_ptr,           # Output [batch_size, 1, seq_len]
    hidden_dim: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that fuses:
    - Type conversions (eliminating redundant ones)
    - Expansion operations (eliminating redundant ones)  
    - Device transfer handling (eliminating unnecessary transfers)
    """
    # Determine which elements this program processes
    hidden_idx = tl.program_id(0) if tl.program_id(1) == 0 else 0
    batch_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Mask bounds checking
    hidden_mask = hidden_idx < hidden_dim
    batch_mask = batch_idx < batch_size
    seq_mask = seq_idx < seq_len
    
    # Process first tensor: expansion and type conversion in one step
    if hidden_idx < hidden_dim and batch_idx == 0:  # Only need to do once per hidden dimension
        # Directly load appropriate type instead of converting twice
        input_val = tl.load(tensor1_ptr + hidden_idx, mask=hidden_mask, other=0.0)
        output1_val = input_val  # Already correct type, eliminate redundant float() calls
        output1_pos = hidden_idx
        tl.store(output1_ptr + output1_pos, output1_val, mask=hidden_mask)
    
    # Process second tensor with expansion and type conversion  
    if batch_idx < batch_size and seq_idx < seq_len:
        input_val = tl.load(tensor2_ptr + batch_idx * seq_len + seq_idx, 
                           mask=batch_mask and seq_mask, other=0.0)
        output2_val = input_val  # Eliminate redundant float() calls
        output2_pos = batch_idx * seq_len + seq_idx  # Store in original format
        tl.store(output2_ptr + output2_pos, output2_val, mask=batch_mask and seq_mask)

@torch.fx.wrap  
def optimized_type_conversion(in_1, in_3):
    """
    Optimized wrapper that eliminates redundant type conversions and device transfers
    """
    # Get tensor information
    hidden_dim = in_1.shape[0] if len(in_1.shape) == 1 else in_1.shape[-1]
    batch_size = in_3.shape[0] 
    seq_len = in_3.shape[1]
    
    # Create output tensors with correct shapes and dtypes
    # Eliminate redundant type conversions - use original dtypes where possible
    output1_shape = (1, hidden_dim, 1)
    output2_shape = (batch_size, 1, seq_len)
    
    # Determine output dtypes based on original (eliminate redundant conversions)
    out1_dtype = in_1.dtype if in_1.dtype == torch.float32 else torch.float32
    out2_dtype = in_3.dtype if in_3.dtype == torch.float32 else torch.float32
    
    output1 = torch.empty(output1_shape, dtype=out1_dtype, device=in_1.device)
    output2 = torch.empty(output2_shape, dtype=out2_dtype, device=in_3.device)
    
    # Handle first tensor expansion (1, hidden_dim, 1)
    # This is a simple expansion that can be done efficiently
    if len(in_1.shape) == 1:
        # For 1D tensor: [hidden_dim] -> [1, hidden_dim, 1]
        output1[:, :, 0] = in_1
    else:
        # For higher dimensions, extract last dimension and expand
        output1[:, :, 0] = in_1[..., -1:] if len(in_1.shape) > 1 else in_1
    
    # Handle second tensor expansion (batch_size, 1, seq_len) 
    # This is a simple expansion that can be done efficiently
    output2[:, 0, :] = in_3
    
    # Note: The redundant device transfers are automatically eliminated
    # since inputs and outputs are already on the same device
    
    return output1, output2

# Alternative optimized version using Triton for complex cases
@torch.fx.wrap
def optimized_type_conversion_triton(in_1, in_3):
    """
    Optimized wrapper using Triton for maximum performance
    """
    # Get tensor information
    hidden_dim = in_1.shape[0]  # Assuming tensor1 is [hidden_dim]
    batch_size = in_3.shape[0] 
    seq_len = in_3.shape[1]
    
    # Create output tensors
    output1 = torch.empty((1, hidden_dim, 1), dtype=torch.float32, device=in_1.device)
    output2 = torch.empty((batch_size, 1, seq_len), dtype=torch.float32, device=in_3.device)
    
    # Grid configuration
    BLOCK_SIZE = 256
    num_hidden = (hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_batch = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE  
    num_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized Triton kernel if tensors are large enough
    if hidden_dim > 1024 or (batch_size * seq_len) > 1024:
        type_conversion_kernel[(num_hidden, num_batch, num_seq)](
            tensor1_ptr=in_1,
            tensor2_ptr=in_3,
            output1_ptr=output1,
            output2_ptr=output2,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            seq_len=seq_len,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # For small tensors, use simpler Python operations
        if len(in_1.shape) == 1:
            output1[:, :, 0] = in_1.to(torch.float32)
        else:
            output1[:, :, 0] = in_1.to(torch.float32)[..., -1:] if len(in_1.shape) > 1 else in_1.to(torch.float32)
        
        output2[:, 0, :] = in_3.to(torch.float32)
    
    return output1, output2

# Replacement function  
def replacement_func():
    return optimized_type_conversion_triton