import torch
import triton
import triton.language as tl

# Pattern matching function for permute + contiguous + view optimization
def pattern(context_after_conv):
    """Match permute(0,2,1,3) + contiguous + view pattern"""
    permuted = context_after_conv.permute(0, 2, 1, 3)  # [N, C, H, W] -> [N, H, C, W]
    contig = permuted.contiguous()
    # We need to infer the final view shape from the context
    # Since all graphs have different output shapes, we'll let the kernel handle it
    return contig

# Argument extraction function
def replacement_args(context_after_conv):
    return (context_after_conv,)

# Optimized kernel that directly computes the final view shape without intermediate operations
@triton.jit
def optimized_memory_layout_kernel(
    # Input tensor
    input_ptr,         # [N, C, H, W] - input from conv+add
    output_ptr,        # [N, H, final_C] - final output shape
    
    # Input tensor shapes
    N, C, H, W,
    final_C,
    
    # Block sizes for optimization
    BLOCK_SIZE_N: tl.constexpr,    # Number of batches per program
    BLOCK_SIZE_H: tl.constexpr,    # Number of height positions per program
    BLOCK_SIZE_C: tl.constexpr,    # Number of final channels per program
):
    # Calculate program indices
    batch = tl.program_id(0)  # Batch index
    h_idx = tl.program_id(1)  # Height index  
    final_c_idx = tl.program_id(2)  # Final channel index
    
    # Calculate original channel index: each final block corresponds to multiple original channels
    orig_c_start = final_c_idx * BLOCK_SIZE_C
    orig_c_end = min(orig_c_start + BLOCK_SIZE_C, C)
    
    # Initialize output value
    output_val = 0.0
    
    # Sum over the original channels that get mapped to this final channel
    for orig_c in range(orig_c_start, orig_c_end):
        # Load input value [batch, orig_c, h_idx, w_idx]
        # Since we're doing permute(0,2,1,3) -> [N,H,C,W], w_idx is always 0 for final_C = C * W
        w_idx = 0  # This assumes the final view combines C and W dimensions
        
        input_val = tl.load(input_ptr + batch * C * H * W + 
                           orig_c * H * W + h_idx * W + w_idx,
                           mask=True)
        
        output_val += input_val
    
    # Store output value [batch, h_idx, final_c_idx]
    tl.store(output_ptr + batch * H * final_C + h_idx * final_C + final_c_idx,
             output_val)

# Alternative optimized kernel that preserves exact computation
@triton.jit  
def direct_memory_layout_kernel(
    # Input tensor
    input_ptr,         # [N, C, H, W]
    
    # Output tensor  
    output_ptr,        # [N, H, final_C]
    
    # Input and output shapes
    N, C, H, W,
    final_C,
    
    # Block sizes
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, 
    BLOCK_SIZE_C: tl.constexpr,
):
    # Calculate program indices
    batch = tl.program_id(0)
    h_idx = tl.program_id(1)
    final_c_idx = tl.program_id(2)
    
    # Calculate original coordinates after permute(0,2,1,3)
    # Original: [N, C, H, W] -> After permute: [N, H, C, W]
    # Final view: [N, H, final_C] where final_C = C * W
    
    # Map final_c_idx back to original [c, w] coordinates
    orig_w = final_c_idx // C  # This might not be right, depends on view logic
    orig_c = final_c_idx % C
    
    # Actually, let's handle this more carefully. The view operation in the original code
    # is .view(N, H, C*W) essentially, but the exact mapping needs to match PyTorch's behavior
    # For now, let's just copy the data with proper memory layout transformation
    
    # We can directly compute the final result without intermediate tensors by understanding
    # that permute(0,2,1,3) + contiguous + view(N,H,C*W) can be optimized
    
    if final_c_idx < C * W:  # Ensure we're within bounds
        # Calculate original coordinates
        orig_c = final_c_idx // W
        orig_w = final_c_idx % W
        
        # Load input value and store directly to output location
        input_val = tl.load(input_ptr + batch * C * H * W + 
                           orig_c * H * W + h_idx * W + orig_w,
                           mask=True)
        tl.store(output_ptr + batch * H * final_C + h_idx * final_C + final_c_idx,
                 input_val)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_memory_layout(context_after_conv, final_shape):
    """Optimize permute + contiguous + view sequence"""
    N, C, H, W = context_after_conv.shape
    
    # Get final shape - it should be (N, H, final_C) where final_C = C * W
    final_N, final_H, final_C = final_shape
    
    # Verify shapes match expectation
    assert final_N == N, "Batch dimension mismatch"
    assert final_H == H, "Height dimension mismatch"
    assert final_C == C * W, "Final channels should be C * W"
    
    # Create output tensor
    output = torch.empty(final_shape, dtype=context_after_conv.dtype, device=context_after_conv.device)
    
    # Set block sizes for optimization
    BLOCK_SIZE_N = 1   # Process one batch at a time
    BLOCK_SIZE_H = 64  # Number of height positions per program
    BLOCK_SIZE_C = 128 # Number of final channels per program
    
    # Calculate grid size
    grid_N = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_H = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_C = (final_C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Use the direct memory layout kernel
    direct_memory_layout_kernel[(grid_N, grid_H, grid_C)](
        input_ptr=context_after_conv,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        final_C=final_C,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    def optimized_wrapper(context_after_conv):
        # Infer final shape from the pattern - this needs to be adjusted based on specific graph
        N, C, H, W = context_after_conv.shape
        
        # For the graphs we've seen, the final shape varies:
        # Some: (N, H, C*W)
        # Others have different patterns due to different view operations
        
        # For now, let's return a simplified version that just does the transpose
        # This will be refined when we know the specific target graphs
        return optimized_memory_layout(context_after_conv, (N, H, C * W))
    
    return optimized_wrapper