import torch
import triton
import triton.language as tl

def pattern(tmp_8):
    # AdaptiveAvgPool2d to 1x1 is equivalent to computing spatial mean
    tmp_9 = torch.nn.functional.adaptive_avg_pool2d(tmp_8, 1)
    # Flatten from channel dimension
    tmp_10 = tmp_9.flatten(1, -1)
    # Return the result (this is what's observable outside)
    return tmp_10

def replacement_args(tmp_8):
    return (tmp_8,)

@triton.jit
def spatial_mean_kernel(
    input_ptr,      # Input tensor [N, C, H, W]
    output_ptr,     # Output [N, C] 
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Calculate program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Calculate ranges for each program
    n_start = pid_n * BLOCK_SIZE_N
    c_start = pid_c * BLOCK_SIZE_C
    
    # Create offsets
    n_offset = n_start + tl.arange(0, BLOCK_SIZE_N)
    c_offset = c_start + tl.arange(0, BLOCK_SIZE_C)
    
    # Create masks for bounds checking
    n_mask = n_offset < N
    c_mask = c_offset < C
    
    # Compute total spatial elements for division
    total_spatial = H * W
    inv_spatial = 1.0 / total_spatial
    
    # Process multiple n, c combinations per program
    n_idx = n_offset[:, None]
    c_idx = c_offset[None, :]
    
    # Create output indices
    output_offsets = n_idx * C + c_idx
    
    # Accumulate sum spatially (this is simplified - would need more complex accumulation for full generality)
    # For this specific pattern where we know the dimensions, we can optimize
    if H == 8 and W == 8:
        # Special case for 8x8 spatial dimensions (common in the input examples)
        spatial_sum = 0.0
        
        # Load all spatial elements and sum them up
        # This is a simplified approach for demonstration
        # In practice, you'd want to use more efficient accumulation patterns
        
        # Create spatial indices
        h_idx = tl.arange(0, H)
        w_idx = tl.arange(0, W)
        h_grid, w_grid = tl.meshgrid(h_idx, w_idx)
        
        # Calculate input offsets: n * C * H * W + c * H * W + h * W + w
        for h_val in range(H):
            for w_val in range(W):
                input_offset_calc = n_idx * C * H * W + c_idx * H * W + h_val * W + w_val
                val = tl.load(input_ptr + input_offset_calc, mask=n_mask[:, None] & c_mask[None, :], other=0.0)
                spatial_sum += val
        
        # Compute mean by dividing by number of spatial elements
        mean_val = spatial_sum * inv_spatial
        
        # Store the result
        tl.store(output_ptr + output_offsets, mean_val, mask=n_mask[:, None] & c_mask[None, :])
    
    elif H == 7 and W == 7:
        # Special case for 7x7 spatial dimensions 
        spatial_sum = 0.0
        inv_spatial = 1.0 / 49.0  # 7*7
        
        for h_val in range(H):
            for w_val in range(W):
                input_offset_calc = n_idx * C * H * W + c_idx * H * W + h_val * W + w_val
                val = tl.load(input_ptr + input_offset_calc, mask=n_mask[:, None] & c_mask[None, :], other=0.0)
                spatial_sum += val
        
        mean_val = spatial_sum * inv_spatial
        tl.store(output_ptr + output_offsets, mean_val, mask=n_mask[:, None] & c_mask[None, :])
    
    else:
        # Generic case - use a more sophisticated accumulation approach  
        # For now, skip generic implementation and focus on specific sizes
        # This would need proper implementation for general case
        pass

@torch.fx.wrap
def optimized_spatial_mean(input_tensor):
    # Get tensor shapes
    N, C, H, W = input_tensor.shape
    
    # Set block sizes for optimal GPU occupancy
    BLOCK_SIZE_N = 4   # Process multiple batch elements
    BLOCK_SIZE_C = 64  # Process multiple channels
    
    # Calculate grid size
    num_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Create output tensor [N, C]
    output = torch.empty((N, C), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # For the specific patterns we see, we can optimize better
    # Use specialized kernel for common sizes
    if H == 8 and W == 8:
        # Optimized kernel for 8x8 spatial size
        @triton.jit
        def optimized_8x8_kernel(
            input_ptr, output_ptr, N, C, H, W,
            BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
        ):
            pid_n = tl.program_id(0)
            pid_c = tl.program_id(1)
            
            n_start = pid_n * BLOCK_SIZE_N
            c_start = pid_c * BLOCK_SIZE_C
            
            n_offset = n_start + tl.arange(0, BLOCK_SIZE_N)
            c_offset = c_start + tl.arange(0, BLOCK_SIZE_C)
            
            n_mask = n_offset < N
            c_mask = c_offset < C
            
            n_idx = n_offset[:, None]
            c_idx = c_offset[None, :]
            
            # Optimized 8x8 computation - process each spatial location
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=tl.float32)
            
            for h in range(H):
                for w in range(W):
                    input_offset = n_idx * C * H * W + c_idx * H * W + h * W + w
                    val = tl.load(input_ptr + input_offset, mask=n_mask[:, None] & c_mask[None, :], other=0.0)
                    acc += val
            
            # Compute mean (divide by 64 for 8x8)
            mean_val = acc * (1.0 / (H * W))
            
            # Store result
            output_offset = n_idx * C + c_idx
            tl.store(output_ptr + output_offset, mean_val, mask=n_mask[:, None] & c_mask[None, :])
        
        optimized_8x8_kernel[(num_n, num_c)](
            input_tensor, output, N, C, H, W,
            BLOCK_SIZE_N, BLOCK_SIZE_C
        )
        
    elif H == 7 and W == 7:
        # Optimized kernel for 7x7 spatial size
        @triton.jit
        def optimized_7x7_kernel(
            input_ptr, output_ptr, N, C, H, W,
            BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
        ):
            pid_n = tl.program_id(0)
            pid_c = tl.program_id(1)
            
            n_start = pid_n * BLOCK_SIZE_N
            c_start = pid_c * BLOCK_SIZE_C
            
            n_offset = n_start + tl.arange(0, BLOCK_SIZE_N)
            c_offset = c_start + tl.arange(0, BLOCK_SIZE_C)
            
            n_mask = n_offset < N
            c_mask = c_offset < C
            
            n_idx = n_offset[:, None]
            c_idx = c_offset[None, :]
            
            # Optimized 7x7 computation
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=tl.float32)
            
            for h in range(H):
                for w in range(W):
                    input_offset = n_idx * C * H * W + c_idx * H * W + h * W + w
                    val = tl.load(input_ptr + input_offset, mask=n_mask[:, None] & c_mask[None, :], other=0.0)
                    acc += val
            
            # Compute mean (divide by 49 for 7x7)
            mean_val = acc * (1.0 / (H * W))
            
            # Store result
            output_offset = n_idx * C + c_idx
            tl.store(output_ptr + output_offset, mean_val, mask=n_mask[:, None] & c_mask[None, :])
        
        optimized_7x7_kernel[(num_n, num_c)](
            input_tensor, output, N, C, H, W,
            BLOCK_SIZE_N, BLOCK_SIZE_C
        )
    
    else:
        # Fallback to PyTorch for non-standard sizes (this should be optimized in full implementation)
        # For now, return empty tensor - this pass will skip non-standard sizes
        # In a full implementation, you'd want to handle this case properly
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        if H <= 16 and W <= 16:  # Handle small cases
            # Simple fallback for small sizes
            return torch.mean(input_tensor, dim=[2, 3])
        else:
            # For larger sizes, skip optimization and return original computation
            return torch.nn.functional.adaptive_avg_pool2d(input_tensor, 1).flatten(1, -1)
    
    return output

def replacement_func():
    return optimized_spatial_mean