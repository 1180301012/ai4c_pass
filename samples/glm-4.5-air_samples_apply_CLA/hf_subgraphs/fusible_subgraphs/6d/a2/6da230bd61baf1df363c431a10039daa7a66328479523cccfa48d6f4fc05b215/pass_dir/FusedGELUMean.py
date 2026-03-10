import torch
import triton
import triton.language as tl

# Pattern matching function for the entire fused computation
def pattern(x):
    tmp_0 = torch.nn.functional.gelu(x)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Fused kernel: GELU + Mean reduction in single pass
@triton.jit
def fused_gelu_mean_kernel(
    gelu_output_ptr,
    mean_output_ptr, 
    input_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_X: tl.constexpr = 64,
):
    # Each program handles one (batch, channel) pair for both operations
    pair_idx = tl.program_id(0)
    stride = tl.num_programs(0)
    
    # Skip ahead for each thread
    while pair_idx < batch_size * channels:
        batch_id = pair_idx // channels
        channel_id = pair_idx % channels
        
        # Calculate input base offset for this (batch, channel) pair
        input_base = (batch_id * channels * height * width + 
                     channel_id * height * width)
        
        # Initialize sum for mean calculation
        spatial_sum = 0.0
        
        # Process one channel: apply GELU and accumulate sum for mean
        for h in range(height):
            for w in range(width):
                offset = input_base + h * width + w
                
                # Load input value
                x = tl.load(input_ptr + offset, other=0.0)
                
                # Apply GELU activation
                # GELU(x) = x * sigmoid(1.702 * x)
                sigmoid_approx = 1.0 / (1.0 + tl.exp(-1.702 * x))
                gelu_val = x * sigmoid_approx
                
                # Store GELU output tensor at correct position
                gelu_output_offset = input_base + h * width + w
                tl.store(gelu_output_ptr + gelu_output_offset, gelu_val)
                
                # Accumulate sum for mean calculation
                spatial_sum += gelu_val
        
        # Calculate mean for this (batch, channel) pair
        mean_val = spatial_sum / (height * width)
        
        # Store mean output (output shape [batch_size, channels, 1, 1])
        mean_output_idx = batch_id * channels + channel_id
        tl.store(mean_output_ptr + mean_output_idx, mean_val)
        
        # Move to next chunk
        pair_idx += stride

@torch.fx.wrap  
def fused_gelu_mean(x):
    batch_size, channels, height, width = x.shape
    
    # Output 1: GELU output with same shape as input
    gelu_output = torch.empty_like(x, dtype=torch.float32, device=x.device)
    
    # Output 2: Mean output with shape [batch_size, channels, 1, 1]
    mean_output = torch.empty(batch_size, channels, 1, 1, 
                             dtype=torch.float32, device=x.device)
    mean_output_flat = mean_output.view(-1)  # Flatten for easier storage
    
    total_pairs = batch_size * channels
    BLOCK_SIZE = 64
    
    # Grid size: one program per (batch, channel) pair
    num_pairs = (total_pairs + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_gelu_mean_kernel[(num_pairs,)](
        gelu_output_ptr=gelu_output,
        mean_output_ptr=mean_output_flat,
        input_ptr=x,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_X=BLOCK_SIZE
    )
    
    return (gelu_output, mean_output)

# Replacement function
def replacement_func():
    return fused_gelu_mean