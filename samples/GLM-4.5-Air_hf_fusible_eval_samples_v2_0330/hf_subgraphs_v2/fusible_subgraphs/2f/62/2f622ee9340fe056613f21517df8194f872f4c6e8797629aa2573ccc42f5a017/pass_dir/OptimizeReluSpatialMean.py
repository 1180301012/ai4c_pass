import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the ReLU and spatial mean pattern
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_3

def replacement_args(in_0, in_1):
    # We only need the tensor input for this optimization
    return (None, in_1)

@triton.jit
def optimized_relu_mean_kernel(
    input_ptr,
    relu_out_ptr,
    mean_out_ptr,
    batch_size,
    num_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Grid setup: each block handles a subset of channels
    pid = tl.program_id(0)
    num_programs = tl.cdiv(num_channels, BLOCK_SIZE_M)
    
    # Initialize local sums for mean calculation
    local_sum = 0.0
    
    # Iterate over the block of channels assigned to this program
    start_channel = pid * BLOCK_SIZE_M
    for m in range(BLOCK_SIZE_M):
        channel = start_channel + m
        if channel >= num_channels:
            break
            
        # Iterate over spatial dimensions
        for h in range(height):
            for w in range(width):
                # Compute the offset
                offset = (batch_size * num_channels * height * width + 
                         channel * height * width + 
                         h * width + w)
                
                # Load input value
                x = tl.load(input_ptr + offset, other=0.0)
                # Apply ReLU
                relu_val = tl.maximum(x, 0.0)
                # Store ReLU result
                tl.store(relu_out_ptr + offset, relu_val)
                # Accumulate for mean
                local_sum += float(relu_val)
    
    # Compute mean for this channel block
    spatial_area = float(height * width)
    if BLOCK_SIZE_M > 0:
        mean_val = local_sum / (spatial_area * min(BLOCK_SIZE_M, num_channels - start_channel))
    else:
        mean_val = 0.0
    
    # Store mean result
    mean_offset = batch_size * num_channels + start_channel
    tl.store(mean_out_ptr + mean_offset, mean_val)

@torch.fx.wrap
def optimized_relu_mean(in_1):
    batch_size, num_channels, height, width = in_1.shape
    
    # Create output tensors
    relu_out = torch.empty_like(in_1)
    mean_out = torch.empty((batch_size, num_channels, 1, 1), dtype=in_1.dtype, device=in_1.device)
    
    # Flatten the spatial mean output for easier kernel access
    mean_flat = mean_out.view(batch_size * num_channels)
    
    # Set up grid and launch kernel
    BLOCK_SIZE_M = 64  # Number of channels per block
    num_programs = tl.cdiv(num_channels, BLOCK_SIZE_M)
    
    optimized_relu_mean_kernel[(num_programs,)](
        input_ptr=in_1.data_ptr(),
        relu_out_ptr=relu_out.data_ptr(),
        mean_out_ptr=mean_flat.data_ptr(),
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=1,
    )
    
    return relu_out, mean_out

def replacement_func():
    return optimized_relu_mean