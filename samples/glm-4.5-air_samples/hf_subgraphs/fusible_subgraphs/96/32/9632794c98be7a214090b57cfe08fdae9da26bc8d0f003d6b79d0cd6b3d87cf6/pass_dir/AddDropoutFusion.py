import torch
import triton
import triton.language as tl
import math

# Pattern matching function for addition + dropout fusion
def pattern(in_4, in_3):
    # This matches the addition + dropout operations in the computation graph
    # tmp_3 = in_4 + in_3
    # tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, training=False, inplace=False)
    return tmp_3, tmp_4

# Argument extraction function
def replacement_args(in_4, in_3):
    return (in_4, in_3)

# Helper function to get dropout mask
@triton.jit
def dropout_mask_kernel(
    mask_ptr,
    height, width, batch_size, channels,
    prob: tl.constexpr, seed: tl.constexpr
):
    pid = tl.program_id(0)
    mask_size = height * width
    
    offset = pid * mask_size + tl.arange(0, mask_size)
    
    # Generate random numbers using hash function
    rand_vals = tl.rand(seed, offset) 
    mask = (rand_vals > prob).to(tl.float32)
    
    tl.store(mask_ptr + offset, mask, mask=(pid < batch_size * channels))

# Optimized kernel for fusion of addition + dropout
@triton.jit
def add_dropout_fusion_kernel(
    x1_ptr, x2_ptr, out_ptr, tmp_3_ptr,
    batch_size, channels, height, width,
    prob: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < batch_size * channels * height * width
    
    # Load both input tensors (flatten batch and channels)
    offset = pid
    x1 = tl.load(x1_ptr + offset, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offset, mask=mask, other=0.0)
    
    # Perform addition
    tmp_3_vals = x1 + x2
    
    # Compute dropout mask on the fly using simple deterministic approach
    # Using a hash-based approach to generate pseudo-random values
    rand_seed = (pid * 123456789) & 0xFFFFFFFF
    rand_val = (rand_seed * 1103515245 + 12345) & 0x7FFFFFFF
    normalized_rand = rand_val / 2147483648.0
    
    dropout_mask = (normalized_rand > prob).to(tl.float32) * (1.0 / (1.0 - prob))
    
    # Apply dropout
    out = tmp_3_vals * dropout_mask
    
    # Store both intermediate result (tmp_3) and final result (tmp_4)
    tl.store(tmp_3_ptr + offset, tmp_3_vals, mask=mask)
    tl.store(out_ptr + offset, out, mask=mask)

@torch.fx.wrap
def optimized_add_dropout_fusion(in_4, in_3):
    # Get tensor dimensions
    batch_size, channels, height, width = in_4.shape
    total_elements = batch_size * channels * height * width
    
    # Create output tensors
    tmp_3_out = torch.empty_like(in_4)
    out = torch.empty_like(in_4)
    
    # Optimize block size based on tensor size
    if total_elements >= 65536:  # Large tensor
        BLOCK_SIZE = 1024
    elif total_elements >= 16384:  # Medium tensor
        BLOCK_SIZE = 512
    else:  # Small tensor
        BLOCK_SIZE = 256
    
    # Calculate grid dimensions
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    add_dropout_fusion_kernel[(num_programs,)](
        in_4, in_3, out, tmp_3_out,
        batch_size, channels, height, width,
        0.1, BLOCK_SIZE
    )
    
    return tmp_3_out, out

def replacement_func():
    return optimized_add_dropout_fusion