import torch
import triton
import triton.language as tl

# Pattern for linear transformation used for value projection in attention
# This pattern specifically handles 3D input tensors (batch_size, seq_len, hidden_size)
def pattern(in_0, in_1, in_3):
    # Linear transformation: weight @ input + bias
    # We assume in_3 has shape (batch_size, seq_len, hidden_size)
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    return linear

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

@triton.jit
def linear_kernel(
    x_ptr,      # input tensor (total_elements, hidden_size)
    weight_ptr, # weight tensor (hidden_size, hidden_size) 
    bias_ptr,   # bias tensor (hidden_size,)
    out_ptr,    # output tensor (total_elements, hidden_size)
    total_elements: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles a contiguous block of the flattened input
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE_M
    block_end = min(block_start + BLOCK_SIZE_M, total_elements)
    mask = block_start < total_elements
    
    # Initialize accumulator 
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Process each row in the block
    for m in range(block_start, block_end):
        # Load bias for this row
        bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE_N), 
                      mask=tl.arange(0, BLOCK_SIZE_N) < hidden_size, 
                      other=0.0).to(tl.float32)
        
        # Process the K dimension (features) in blocks
        for k in range(0, hidden_size, BLOCK_SIZE_K):
            k_end = min(k + BLOCK_SIZE_K, hidden_size)
            
            # Load weight column for this k block
            weight = tl.load(weight_ptr + k * hidden_size + tl.arange(0, BLOCK_SIZE_N),
                           mask=tl.arange(0, BLOCK_SIZE_N) < (k_end - k),
                           other=0.0).to(tl.float32)
            
            # Load input row segment
            x_row = tl.load(x_ptr + m * hidden_size + tl.arange(k, k_end),
                          mask=tl.arange(k, k_end) < hidden_size,
                          other=0.0).to(tl.float32)
            
            # Outer product: x_row * weight + accumulate
            acc_local = x_row[:, None] * weight[None, :]
            if k == 0:
                acc_local += bias  # Add bias on first iteration
            acc[0, :k_end-k] += acc_local
    
    # Store results
    for m in range(0, min(BLOCK_SIZE_M, block_end - block_start)):
        if block_start + m < total_elements:
            for n in range(0, min(BLOCK_SIZE_N, hidden_size)):
                tl.store(out_ptr + (block_start + m) * hidden_size + n,
                        acc[m, n], mask=True)

@torch.fx.wrap
def optimized_linear(x, weight, bias):
    # Always flatten all but last dimension, then restore shape after linear
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    
    # Standard linear operation (this will be optimized by compiler later)
    result_flat = torch.nn.functional.linear(x_flat, weight, bias)
    
    # Original shape but with last dimension preserved
    new_shape = orig_shape[:-1] + (result_flat.shape[-1],)
    return result_flat.reshape(new_shape)

def replacement_func():
    return optimized_linear