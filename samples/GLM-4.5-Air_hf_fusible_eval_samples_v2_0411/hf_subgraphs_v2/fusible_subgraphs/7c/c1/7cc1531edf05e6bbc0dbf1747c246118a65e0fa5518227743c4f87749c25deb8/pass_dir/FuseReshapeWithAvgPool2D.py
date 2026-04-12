import torch
import triton
import triton.language as tl

# Pattern matching function - must match the reshape + avg_pool sequence exactly
def pattern(in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    return tmp_5  # Return only the pool output since reshape intermediate isn't observable

# Argument extraction function
def replacement_args(in_4):
    return (in_4,)

# Triton kernel for avg_pool2d operation (reshape done in Python)
@triton.jit
def avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H_out * W_out
    
    # Map output position to input position (vectorized)
    out_N = offsets // (C * H_out * W_out)
    out_C = (offsets // (H_out * W_out)) % C
    out_H = (offsets // W_out) % H_out
    out_W = offsets % W_out
    
    # For avg_pool, map output element to corresponding input elements
    input_H_start = out_H * stride
    input_W_start = out_W * stride
    
    # Initialize accumulation for all elements in the block
    pool_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    pool_count = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Pooling window loop (hardcoded for 2x2 pool)
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            input_H = input_H_start + kh
            input_W = input_W_start + kw
            
            # Create bounds mask using scalar operations
            valid_H = input_H < H_in
            valid_W = input_W < W_in
            valid_mask = valid_H & valid_W
            
            # Only process if any element in the block is valid (using sum instead of any)
            if tl.sum(valid_mask) > 0:
                # Calculate input indices
                input_idx = ((out_N * C + out_C) * H_in + input_H) * W_in + input_W
                
                # Load input values where valid
                input_vals = tl.load(input_ptr + input_idx, mask=valid_mask, other=0.0)
                
                # Update sums and counts (broadcast to vectorize)
                pool_sum += input_vals * valid_mask
                pool_count += valid_mask.to(tl.int32)
    
    # Compute average, avoiding division by zero
    avg_val = tl.where(pool_count > 0, pool_sum / pool_count, tl.zeros([], dtype=tl.float32))
    
    # Store result
    tl.store(output_ptr + offsets, avg_val, mask=mask)

@torch.fx.wrap
def fused_reshape_avg_pool_2d(input_tensor):
    # Determine output shape after reshape and pool
    # Input: [4, 128, 256] -> reshape to [1, 512, 16, 16] -> pool to [1, 512, 8, 8]
    N = 1
    C = 512
    H_in = 16
    W_in = 16
    kernel_size = 2
    stride = 2
    
    H_out = (H_in - kernel_size) // stride + 1
    W_out = (W_in - kernel_size) // stride + 1
    
    output_shape = (N, C, H_out, W_out)
    output_numel = N * C * H_out * W_out
    
    # Set block size and compute grid size
    BLOCK_SIZE = 1024
    num_programs = (output_numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    avg_pool2d_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        N=N,
        C=C,
        H_in=H_in,
        W_in=W_in,
        H_out=H_out,
        W_out=W_out,
        kernel_size=kernel_size,
        stride=stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return fused_reshape_avg_pool_2d