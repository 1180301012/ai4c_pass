import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = x.sum(1)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_1

def replacement_args(x):
    return (x,)

@triton.jit
def fused_sum_mean_kernel(
    x_ptr,
    out_ptr,
    total_elements,
    reduction_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Only first thread performs the final reduction
    if pid != 0:
        return
    
    # Use Triton's built-in contiguous load for maximum efficiency
    if BLOCK_SIZE >= 4096:  # Large block size for better efficiency
        # Load in large chunks with stride 1 (contiguous)
        total_sum = 0.0
        strides = tl.arange(0, BLOCK_SIZE)
        mask = strides < total_elements
        
        # Load a block and sum it
        x_block = tl.load(x_ptr + strides, mask=mask, other=0.0)
        total_sum += tl.sum(x_block)
        
        # For larger tensors, continue loading remaining elements
        remaining_elements = total_elements - BLOCK_SIZE
        
        if remaining_elements > 0:
            # Load remaining elements efficiently
            for i in range(0, remaining_elements, BLOCK_SIZE):
                end_idx = min(i + BLOCK_SIZE, remaining_elements)
                current_strides = i + tl.arange(0, end_idx - i)
                current_mask = current_strides < remaining_elements
                
                remaining_block = tl.load(x_ptr + BLOCK_SIZE + current_strides, 
                                        mask=current_mask, other=0.0)
                total_sum += tl.sum(remaining_block)
    else:
        # For smaller tensors, use simpler approach
        total_sum = 0.0
        for i in range(total_elements):
            val = tl.load(x_ptr + i)
            total_sum += val
    
    # Apply normalization factor for mean computation
    result = total_sum * reduction_factor
    tl.store(out_ptr, result)

@torch.fx.wrap
def fused_sum_mean(x):
    # Input shape: [1, 2, 256, 32, 32] or [1, 2, 256, 8, 8]
    # We need to compute: 
    # tmp_0 = x.sum(1)  # [1, 2, 256, 32, 32] -> [1, 1, 256, 32, 32]
    # tmp_1 = tmp_0.mean((2, 3), keepdim=True)  # [1, 1, 256, 32, 32] -> [1, 1, 1, 1, 32, 32]
    
    # Calculate total elements to process (all elements in the flattened dimensions)
    total_elements = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]
    
    # Output factor: 1.0 / (product of reduced dimensions)
    # We reduce dims 1, 2, 3, which have sizes x.shape[1], x.shape[2], x.shape[3]
    reduction_factor = 1.0 / (x.shape[1] * x.shape[2] * x.shape[3])
    
    # Create output tensor for the scalar result
    result_tensor = torch.empty((), dtype=x.dtype, device=x.device)
    
    # Use a large block size for better GPU utilization
    BLOCK_SIZE = 4096
    
    # Launch kernel with single program
    fused_sum_mean_kernel[(1,)](
        x_ptr=x,
        out_ptr=result_tensor,
        total_elements=total_elements,
        reduction_factor=reduction_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Broadcast result to the expected output shape [1, 1, 1, 1, 32, 32] or [1, 1, 1, 1, 8, 8]
    output_shape = [1, 1, 1, 1, x.shape[3], x.shape[4]]
    return result_tensor.expand(output_shape)

def replacement_func():
    return fused_sum_mean