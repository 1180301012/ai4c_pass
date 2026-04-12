import torch
import triton
import triton.language as tl

def pattern(x, weight, mult_factor):
    """Pattern: torch.nn.functional.linear(x, weight, None) followed by element-wise multiplication"""
    linear = torch.nn.functional.linear(x, weight, None)
    result = mult_factor * linear
    return result

def replacement_args(x, weight, mult_factor):
    return (x, weight, mult_factor)

@triton.jit
def fused_linear_mult_kernel(
    x_ptr,
    weight_ptr,
    mult_factor_ptr,
    scalar_mult_factor: tl.constexpr,
    out_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    """Fused linear transformation + element-wise multiplication kernel"""
    # Program identifiers for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of columns each program handles
    cols_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols_mask = cols_offset < out_features
    
    # Range of rows each program handles
    rows_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_mask = rows_offset < batch_size * seq_len
    
    # Determine multiplication factor
    if mult_factor_ptr is not None:
        # Tensor multiplication factor - need to handle broadcasting
        # For simplicity, assume this is a scalar tensor or handle in wrapper
        mult_factor = 1.0  # Default, will be handled in wrapper
    else:
        # Scalar multiplication factor passed directly
        mult_factor = scalar_mult_factor
    
    # Accumulators for output
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension (inner product)
    for k in range(0, in_features, BLOCK_K):
        k_offset = k + tl.arange(0, BLOCK_K)
        k_mask = k_offset < in_features
        
        # Load x tile
        x_tile = tl.load(x_ptr + rows_offset[:, None] * in_features + k_offset[None, :],
                        mask=rows_mask[:, None] & k_mask[None, :],
                        other=0.0).to(tl.float32)
        
        # Load weight tile
        weight_tile = tl.load(weight_ptr + cols_offset[:, None] * in_features + k_offset[None, :],
                             mask=cols_mask[:, None] & k_mask[None, :],
                             other=0.0).to(tl.float32)
        
        # Matrix multiplication fragment
        accumulator += x_tile @ weight_tile
    
    # Apply multiplication factor and store result
    tl.store(out_ptr + rows_offset[:, None] * out_features + cols_offset[None, :], 
             accumulator * mult_factor, 
             mask=rows_mask[:, None] & cols_mask[None, :])

@torch.fx.wrap
def optimized_fused_linear_mult(x, weight, mult_factor):
    """Fused linear transformation + element-wise multiplication using Triton"""
    # Get input tensor shape
    original_shape = x.shape
    if len(original_shape) == 3:
        batch_size, seq_len, in_features = original_shape
        total_elements = batch_size * seq_len
    elif len(original_shape) == 2:
        total_elements, in_features = original_shape
        batch_size = total_elements
        seq_len = 1
    else:
        # 1D tensor - treat as single batch
        total_elements = 1
        in_features = original_shape[0] if len(original_shape) == 1 else 1
        batch_size = 1
        seq_len = 1
    
    out_features = weight.shape[0]
    
    # Set block sizes for optimal performance
    BLOCK_K = 32  # Inner dimension block size
    BLOCK_N = min(128, out_features)  # Output features block size
    BLOCK_M = min(1024, total_elements)  # Batch*Seq block size
    
    # Calculate grid dimensions
    grid_m = (total_elements + BLOCK_M - 1) // BLOCK_M
    grid_n = (out_features + BLOCK_N - 1) // BLOCK_N
    
    # Allocate output tensor in flattened form
    out_flat = torch.empty((total_elements, out_features), dtype=x.dtype, device=x.device)
    
    # Create a constant tensor for mult_factor if it's scalar, otherwise use the tensor pointer
    if isinstance(mult_factor, torch.Tensor):
        # If it's already a tensor, use it directly
        # For simplicity in the kernel, we'll handle this case in kernel logic
        mult_factor_const = mult_factor  # This will be passed as None and handled in kernel
    else:
        # Instead of creating a tensor, we'll handle scalar in kernel logic
        mult_factor_const = None  # Signal scalar mode
    
    # Launch fused kernel with proper scalar handling
    if isinstance(mult_factor, (int, float)):
        # Scalar multiplication factor
        fused_linear_mult_kernel[(grid_m, grid_n)](
            x_ptr=x,
            weight_ptr=weight,
            mult_factor_ptr=None,  # No tensor factor
            scalar_mult_factor=float(mult_factor),  # Pass scalar directly
            out_ptr=out_flat,
            batch_size=total_elements,
            seq_len=seq_len,
            in_features=in_features,
            out_features=out_features,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K
        )
    else:
        # Tensor multiplication factor - for now handle as scalar 1.0 and apply later
        fused_linear_mult_kernel[(grid_m, grid_n)](
            x_ptr=x,
            weight_ptr=weight,
            mult_factor_ptr=None,  # Simplified for now
            scalar_mult_factor=1.0,
            out_ptr=out_flat,
            batch_size=total_elements,
            seq_len=seq_len,
            in_features=in_features,
            out_features=out_features,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K
        )
        # Apply tensor multiplication factor using regular multiplication
        out_flat = mult_factor * out_flat
    
    # Return result based on original shape - flattened for now
    return out_flat

def replacement_func():
    return optimized_fused_linear_mult