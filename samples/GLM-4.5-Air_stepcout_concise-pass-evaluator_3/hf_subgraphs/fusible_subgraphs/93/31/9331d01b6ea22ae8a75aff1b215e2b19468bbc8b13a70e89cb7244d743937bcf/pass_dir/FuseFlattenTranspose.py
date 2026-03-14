import torch
import triton
import triton.language as tl

@triton.jit
def flatten_transpose_kernel(
    input_ptr,          # Input tensor [B, C, H, W]
    output_ptr,         # Output tensor [B, H*W, C]
    B: tl.constexpr,    # Batch size  
    C: tl.constexpr,    # Channels
    H: tl.constexpr,    # Height
    W: tl.constexpr,    # Width
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 2D grid for better memory locality
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate work range for this program
    total_m = B * H
    total_n = C * W
    
    if pid_m >= tl.cdiv(total_m, BLOCK_SIZE_M) or pid_n >= tl.cdiv(total_n, BLOCK_SIZE_N):
        return
    
    # Calculate 2D offsets within our program's tile
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offsets_m < total_m
    mask_n = offsets_n < total_n
    
    # Broadcast to create 2D index space
    offsets_m = offsets_m[:, None]
    offsets_n = offsets_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Convert flat 2D indices to tensor indices [B, C, H, W]
    b = offsets_m // H
    remainder = offsets_m % H
    h = remainder
    c = offsets_n // W
    w = offsets_n % W
    
    # Calculate linear index in input tensor
    input_idx = b * (C * H * W) + c * (H * W) + h * W + w
    
    # Calculate linear index in output tensor [B, H*W, C]
    output_idx = b * (H * W * C) + h * (W * C) + w * C + c
    
    # Load and store with vectorization
    input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + output_idx, input_val, mask=mask)


@torch.fx.wrap
def optimized_flatten_transpose(input_tensor):
    """Optimized fused flatten and transpose operation with 2D tiling"""
    B, C, H, W = input_tensor.shape
    
    # Create output tensor [B, H*W, C]
    output_tensor = torch.empty((B, H*W, C), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use optimal tile sizes based on tensor dimensions
    if H >= 64 and W >= 64:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
    elif H >= 32 and W >= 32:
        BLOCK_SIZE_M = 8
        BLOCK_SIZE_N = 32
    else:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 8
    
    # Calculate 2D grid size for better memory locality
    grid_m = triton.cdiv(B * H, BLOCK_SIZE_M)
    grid_n = triton.cdiv(C * W, BLOCK_SIZE_N)
    grid = (grid_m, grid_n)
    
    try:
        flatten_transpose_kernel[grid](
            input_tensor,
            output_tensor,
            B, C, H, W,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N
        )
    except Exception:
        # Fallback to optimized PyTorch if Triton fails
        # Use reshape for better memory layout than separate flatten+transpose
        if W > 1:
            # Reshape to intermediate layout that matches Triton access pattern
            intermediate = input_tensor.reshape(B, C, H, W)
            transposed = intermediate.transpose(1, 2).reshape(B, H, W, C)
            result = transposed.reshape(B, H*W, C)
        else:
            # Simple case
            flattened = input_tensor.flatten(2)
            transposed = flattened.transpose(1, 2)
            result = transposed
        
        return result
    
    return output_tensor


def pattern(input_tensor):
    """Match flatten(2) -> transpose(1, 2) pattern"""
    flattened = input_tensor.flatten(2)
    transposed = flattened.transpose(1, 2)
    return transposed


def replacement_args(input_tensor):
    """Extract arguments for the replacement function"""
    return (input_tensor,)


def replacement_func():
    """Return the optimized kernel function wrapper"""
    return optimized_flatten_transpose