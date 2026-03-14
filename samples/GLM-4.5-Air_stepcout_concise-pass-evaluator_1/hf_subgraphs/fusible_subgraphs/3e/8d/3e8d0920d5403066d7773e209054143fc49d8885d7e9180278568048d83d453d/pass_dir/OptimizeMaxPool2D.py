import torch
import triton
import triton.language as tl


def pattern(input_in):
    tmp_5 = torch.nn.functional.max_pool2d(input_in, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    return tmp_5


def replacement_args(input_in):
    return (input_in,)


@triton.jit
def optimized_max_pool2d_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    input_height,
    input_width,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each thread block processes a tile of the output
    m = tl.program_id(0)  # height position in output
    n = tl.program_id(1)  # width position in output  
    k = tl.program_id(2)  # channel block
    
    # Calculate output spatial position and bounds
    output_m_base = m * BLOCK_SIZE_M
    output_n_base = n * BLOCK_SIZE_N
    
    # Each thread processes a block of channels
    batch_base = k * BLOCK_SIZE_K
    batch_end = min(batch_base + BLOCK_SIZE_K, n_channels)
    
    # Calculate input coordinates considering padding
    input_start_m = output_m_base * stride - padding
    input_start_n = output_n_base * stride - padding
    
    # Process each channel in the batch
    for c in range(batch_base, batch_end):
        # Process each output pixel in this tile
        for om in range(0, BLOCK_SIZE_M):
            for on in range(0, BLOCK_SIZE_N):
                output_m = output_m_base + om
                output_n = output_n_base + on
                
                # Check bounds
                if output_m < input_height // stride and output_n < input_width // stride:
                    # Calculate 2x2 window bounds in input
                    input_m_start = input_start_m + om * stride
                    input_n_start = input_start_n + on * stride
                    
                    # Initialize max value
                    max_val = -tl.float32('inf')
                    
                    # Process 2x2 window
                    for kw in range(0, kernel_size):
                        for kh in range(0, kernel_size):
                            input_m = input_m_start + kh
                            input_n = input_n_start + kw
                            
                            # Check if input position is valid
                            if (0 <= input_m < input_height and 
                                0 <= input_n < input_width):
                                # Calculate input offset
                                input_offset = (input_m * input_width + input_n) * n_channels + c
                                val = tl.load(input_ptr + input_offset)
                                
                                # Update max
                                if val > max_val:
                                    max_val = val
                    
                    # Calculate output offset and store
                    output_offset = (output_m * (input_width // stride) + output_n) * n_channels + c
                    tl.store(output_ptr + output_offset, max_val)


@torch.fx.wrap
def optimized_max_pool2d(input_in):
    input_shape = input_in.shape
    n_channels, input_height, input_width = input_shape[0], input_shape[2], input_shape[3]
    
    # Create output tensor
    output_height = input_height // 2  # Since stride = 2
    output_width = input_width // 2
    output_shape = [n_channels, output_height, output_width]
    output = torch.empty(output_shape, dtype=input_in.dtype, device=input_in.device)
    
    # Configure block sizes for good GPU utilization
    BLOCK_SIZE_M = 4   # Process 4 output rows per block
    BLOCK_SIZE_N = 4   # Process 4 output columns per block
    BLOCK_SIZE_K = min(256, n_channels)  # Process up to 256 channels per block
    
    # Calculate grid size
    grid_m = (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (n_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid = (grid_m, grid_n, grid_k)
    
    # Launch kernel
    optimized_max_pool2d_kernel[grid](
        input_ptr=input_in,
        output_ptr=output,
        n_channels=n_channels,
        input_height=input_height,
        input_width=input_width,
        kernel_size=2,
        stride=2,
        padding=0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output


def replacement_func():
    return optimized_max_pool2d