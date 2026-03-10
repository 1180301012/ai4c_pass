import torch
import triton
import triton.language as tl

# Pattern matching function - match just the mean operation
def pattern(x):
    # Match just the mean operation pattern  
    return x.mean(-2)

# Argument extraction function  
def replacement_args(bias, weight, input_tensor, sequence_output):
    # Extract arguments needed for the mean operation pattern
    return (sequence_output,)

# Optimized mean kernel for tensors [B, S, F] -> [B, F]
@triton.jit
def mean_kernel(
    x_ptr,           # input tensor [B, S, F]
    y_ptr,           # output tensor [B, F]
    B, S, F,         # dimensions
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr
):
    # Program identifiers for grid computation
    pid_b = tl.program_id(0)
    pid_f = tl.program_id(1)
    
    # Compute ranges for this block
    b_start = pid_b * BLOCK_SIZE_B
    b_end = min((pid_b + 1) * BLOCK_SIZE_B, B)
    f_start = pid_f * BLOCK_SIZE_F
    f_end = min((pid_f + 1) * BLOCK_SIZE_F, F)
    
    # Initialize accumulator for this block
    acc = tl.zeros(b_end - b_start, dtype=tl.float32)
    
    # Compute partial sums for each batch element
    for s in range(S):
        # Load input slice [B_block, F_block]
        x = tl.load(x_ptr + (b_start * S + s) * F + f_start + tl.arange(f_end - f_start),
                   mask=(tl.arange(b_start, b_end) < B)[:, None] & 
                        (f_start + tl.arange(f_end - f_start) < F))
        
        # Sum along the sequence dimension for each batch element
        acc += x.sum(1)
    
    # Divide by sequence length to get mean
    acc = acc / S
    
    # Store result 
    y_offset = b_start * F + f_start
    tl.store(y_ptr + y_offset + tl.arange(b_end - b_start)[:, None] * F + tl.arange(f_end - f_start)[None, :],
             acc, mask=(tl.arange(b_start, b_end) < B)[:, None] & 
                      (f_start + tl.arange(f_end - f_start) < F))

@torch.fx.wrap
def triton_mean(sequence_output):
    B, S, F = sequence_output.shape
    
    # autotune configuration
    grid = lambda meta: (
        (B + meta['BLOCK_SIZE_B'] - 1) // meta['BLOCK_SIZE_B'],
        (F + meta['BLOCK_SIZE_F'] - 1) // meta['BLOCK_SIZE_F']
    )
    
    output = torch.empty((B, F), dtype=sequence_output.dtype, device=sequence_output.device)
    
    # Use efficient block sizes
    mean_kernel[grid](
        sequence_output,
        output,
        B, S, F,
        BLOCK_SIZE_B=32,  # Process 32 batch elements at a time
        BLOCK_SIZE_S=1,   # Process entire sequence at once (inner loop handles S dimension)
        BLOCK_SIZE_F=128  # Process 128 features at a time
    )
    
    return output

def replacement_func():
    return triton_mean