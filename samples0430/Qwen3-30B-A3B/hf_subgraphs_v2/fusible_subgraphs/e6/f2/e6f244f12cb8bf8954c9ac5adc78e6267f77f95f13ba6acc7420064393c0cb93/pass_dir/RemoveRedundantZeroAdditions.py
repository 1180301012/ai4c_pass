import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    tmp_0 = 0 + in_0
    tmp_0 += 0
    tmp_1 = tmp_0
    return tmp_1, tmp_1.mean((2, 3), keepdim=True)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel
def mean_kernel(x_ptr, out_ptr, batch_size, channels, H, W, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    # Compute the mean along spatial dimensions (H and W)
    pid = tl.program_id(0)  # Channel index
    channel = pid
    
    # Each block handles a tile of spatial dimensions
    h = tl.program_id(1)
    w = tl.program_id(2)
    
    start_h = h * BLOCK_H
    start_w = w * BLOCK_W
    
    # Initialize accumulator
    sum_val = tl.zeros((1,), dtype=tl.float32)
    
    # Process all batch elements
    for i in range(batch_size):
        # Load a tile from input
        ptr = x_ptr + (i * channels * H * W) + (channel * H * W) + (start_h * W) + start_w
        x = tl.load(
            ptr,
            shape=(BLOCK_H, BLOCK_W),
            mask=((start_h + tl.arange(0, BLOCK_H)) < H, (start_w + tl.arange(0, BLOCK_W)) < W),
            other=0.0
        )
        sum_val += tl.sum(x)
    
    # Reduce sum across all threads (only the first block stores the result)
    sum_val = tl.sum(sum_val)
    
    if h == 0 and w == 0:
        mean_val = sum_val / (H * W)
        tl.store(out_ptr + channel, mean_val)

# Kernel wrapper
@torch.fx.wrap
def kernel_wrapper(in_0):
    batch_size, channels, H, W = in_0.shape
    
    # Allocate output for mean (will be reshaped to [batch, channels, 1, 1])
    out = torch.empty(channels, dtype=in_0.dtype, device=in_0.device)
    
    # Tuned block sizes for spatial dimensions
    BLOCK_H, BLOCK_W = 16, 16
    num_blocks_h = (H + BLOCK_H - 1) // BLOCK_H
    num_blocks_w = (W + BLOCK_W - 1) // BLOCK_W
    
    # Launch kernel
    mean_kernel[(channels, num_blocks_h, num_blocks_w)](
        in_0,
        out,
        batch_size,
        channels,
        H,
        W,
        BLOCK_H,
        BLOCK_W
    )
    
    # Reshape to [batch, channels, 1, 1] for keepdim=True
    out = out.reshape(1, channels, 1, 1).expand(batch_size, channels, 1, 1)
    
    return (in_0, out)

# Replacement function
def replacement_func():
    return kernel_wrapper