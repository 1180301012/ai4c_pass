import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # AvgPool2D operation with the exact signature from the model
    avg_pool_output = torch.nn.functional.avg_pool2d(input_tensor, 2, 2, 0, True, False, None)
    return input_tensor, avg_pool_output

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def avg_pool2d_kernel(
    input_ptr, output_ptr,
    batch_size, channels, input_h, input_w,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    # Get program IDs
    pid = tl.program_id(0)
    n_offset = pid * BLOCK_SIZE_N
    c_offset = tl.arange(0, BLOCK_SIZE_C)
    
    # Calculate output dimensions (2x2 pool with stride 2)
    output_h = triton.cdiv(input_h, 2)
    output_w = triton.cdiv(input_w, 2)
    
    # Process multiple channels per program for better occupancy
    channel_mask = c_offset < channels
    c_idx = n_offset + c_offset
    
    # Process only valid channels
    if not tl.any(channel_mask):
        return
    
    # Load input tiles for each output position
    for oh in range(output_h):
        for ow in range(output_w):
            # Output position
            output_offset = (
                tl.arange(0, BLOCK_SIZE_C) * (output_h * output_w) + 
                oh * output_w + ow
            )
            
            output_mask = (c_idx < channels) & (oh < output_h) & (ow < output_w)
            sum_val = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
            count = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
            
            # Average over 2x2 window
            for kh in range(2):
                for kw in range(2):
                    ih = oh * 2 + kh
                    iw = ow * 2 + kw
                    
                    # Check if input position is valid
                    input_valid = (ih < input_h) & (iw < input_w)
                    mask = output_mask & input_valid
                    
                    if tl.any(mask):
                        # Calculate input offset
                        input_offset = (
                            c_idx.unsqueeze(0) * (input_h * input_w) + 
                            ih * input_w + iw
                        )
                        
                        # Load input value
                        input_val = tl.load(
                            input_ptr + input_offset, 
                            mask=mask.unsqueeze(0), 
                            other=0.0
                        )
                        
                        # Accumulate sum
                        sum_val = tl.where(mask, sum_val + input_val, sum_val)
                        count = tl.where(mask, count + tl.where(input_valid, 1.0, 0.0), count)
            
            # Compute average (avoid division by zero)
            avg_val = tl.where(count > 0, sum_val / count, sum_val)
            
            # Store output
            output_idx = c_idx * (output_h * output_w) + output_offset
            tl.store(
                output_ptr + output_idx, 
                avg_val, 
                mask=output_mask
            )

@torch.fx.wrap  
def optimized_avg_pool2d(input_tensor):
    batch_size, channels, input_h, input_w = input_tensor.shape
    
    # Calculate output dimensions
    output_h = triton.cdiv(input_h, 2)
    output_w = triton.cdiv(input_w, 2)
    
    # Create output tensor
    output = torch.empty((batch_size, channels, output_h, output_w), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose optimal block sizes based on tensor shapes
    # For better GPU occupancy, we'll process multiple channels per program
    if channels >= 64:
        BLOCK_SIZE_C = 64
    elif channels >= 32:
        BLOCK_SIZE_C = 32
    else:
        BLOCK_SIZE_C = min(channels, 16)
    
    # Calculate number of programs needed
    n_elements = batch_size * channels
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE_C),)
    
    # Launch kernel
    avg_pool2d_kernel[grid_size](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        input_h=input_h,
        input_w=input_w,
        BLOCK_SIZE_N=1,  # Process batch_size sequentially
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return input_tensor, output

def replacement_func():
    return optimized_avg_pool2d