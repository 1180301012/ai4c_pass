import torch
import triton
import triton.language as tl

# Pattern matching function for GELU + Mean reduction fusion
def pattern(x):
    gelu_out = torch.nn.functional.gelu(x)
    mean_out = gelu_out.mean((2, 3), keepdim=True)
    return gelu_out, mean_out

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized fused kernel implementation
@triton.jit
def fused_gelu_mean_kernel(
    in_ptr,
    out_gelu_ptr,
    out_mean_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial location across all batches and channels
    pid = tl.program_id(0)
    
    # Handle batch and channel dimensions
    batch_id = pid // (channels * (height * width // BLOCK_SIZE))
    channel_id = (pid // (height * width // BLOCK_SIZE)) % channels
    spatial_id = (pid % (height * width // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Spatial coordinates
    h = spatial_id // width
    w = spatial_id % width
    
    # Mask to ensure we don't go out of bounds
    mask = (h < height) & (w < width)
    
    # Load input tensor: [batch_size, channels, height, width]
    in_val = tl.load(in_ptr + batch_id * channels * height * width + 
                     channel_id * height * width + h * width + w, 
                     mask=mask, other=0.0)
    
    # Apply GELU activation
    gelu_val = 0.5 * in_val * (1.0 + tl.tanh(0.7978845608028654 * (in_val + 0.044715 * in_val * in_val * in_val)))
    
    # Store GELU output
    tl.store(out_gelu_ptr + batch_id * channels * height * width + 
             channel_id * height * width + h * width + w, 
             gelu_val, mask=mask)
    
    # We'll need to compute the mean separately since it requires accumulation
    # This kernel handles spatial GELU computation

@triton.jit
def compute_final_mean_kernel(
    gelu_ptr,
    out_mean_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch and channel combination
    pid = tl.program_id(0)
    batch_id = pid // channels
    channel_id = pid % channels
    
    # Compute mean by summing over spatial dimensions
    spatial_sum = 0.0
    for h in range(0, height, BLOCK_SIZE):
        for w in range(0, width, BLOCK_SIZE):
            # Sum a BLOCK_SIZE x BLOCK_SIZE region
            block_sum = 0.0
            for bh in range(min(BLOCK_SIZE, height - h)):
                for bw in range(min(BLOCK_SIZE, width - w)):
                    idx = batch_id * channels * height * width + channel_id * height * width + (h + bh) * width + (w + bw)
                    val = tl.load(gelu_ptr + idx, other=0.0)
                    block_sum += val
            spatial_sum += block_sum
    
    # Convert sum to mean
    mean_val = spatial_sum / (height * width)
    
    # Store mean result
    mean_idx = batch_id * channels + channel_id
    tl.store(out_mean_ptr + mean_idx, mean_val)

# Define optimal block sizes
SPATIAL_BLOCK_SIZE = 1024
MEAN_BLOCK_SIZE = 64

# Main wrapper function for the fused computation
@torch.fx.wrap
def fused_gelu_mean_forward(in_0):
    batch_size, channels, height, width = in_0.shape
    
    # Create output tensors
    out_gelu = torch.empty_like(in_0, device=in_0.device)
    out_mean = torch.empty(batch_size, channels, 1, 1, device=in_0.device, dtype=in_0.dtype)
    
    # Launch spatial GELU kernel
    spatial_elements = (batch_size * channels * height * width + SPATIAL_BLOCK_SIZE - 1) // SPATIAL_BLOCK_SIZE
    fused_gelu_mean_kernel[(spatial_elements,)](
        in_ptr=in_0,
        out_gelu_ptr=out_gelu,
        out_mean_ptr=out_mean,  # Will be updated later
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=SPATIAL_BLOCK_SIZE,
    )
    
    # Launch mean computation kernel
    mean_elements = (batch_size * channels + MEAN_BLOCK_SIZE - 1) // MEAN_BLOCK_SIZE
    compute_final_mean_kernel[(mean_elements,)](
        gelu_ptr=out_gelu,
        out_mean_ptr=out_mean,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=MEAN_BLOCK_SIZE,
    )
    
    return out_gelu, out_mean

# Replacement function (returns the optimized function)
def replacement_func():
    return fused_gelu_mean_forward