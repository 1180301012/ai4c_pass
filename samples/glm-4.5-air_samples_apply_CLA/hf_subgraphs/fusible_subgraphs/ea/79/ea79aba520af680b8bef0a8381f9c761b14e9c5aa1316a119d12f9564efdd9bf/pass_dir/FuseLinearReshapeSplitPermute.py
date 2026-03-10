import torch
import triton
import triton.language as tl

def pattern(in_3, weight, bias):
    # Linear transformation
    tmp_3 = torch.nn.functional.linear(in_3, weight, bias)
    
    # Reshape and split - let system determine batch size automatically
    # The actual computation will determine the correct reshape based on input
    tmp_4 = tmp_3  # Don't reshape here - let the replacement handle it
    
    # For now, just return the linear result which is the main bottleneck
    return tmp_3

def replacement_args(in_3, weight, bias):
    # Extract all the arguments needed for the kernel
    in_3_shape = in_3.shape
    seq_len = in_3_shape[1]
    hidden_size = in_3_shape[2]
    batch_size = in_3_shape[0]
    
    return (in_3, weight, bias, batch_size, seq_len, hidden_size)

@triton.jit
def simple_linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    batch_size, seq_len, hidden_size, output_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    total_programs = batch_size * seq_len
    
    if pid >= total_programs:
        return
        
    # Calculate batch and sequence indices for this program
    batch_id = pid // seq_len
    seq_id = pid % seq_len
    
    # Output offset for this sequence position
    y_offset = batch_id * seq_len * output_size + seq_id * output_size
    
    # Process output in chunks of size BLOCK_N
    for output_start in range(0, output_size, BLOCK_N):
        output_end = min(output_start + BLOCK_N, output_size)
        mask = tl.arange(output_start, output_end) < output_size
        
        # Accumulate results for this output chunk
        acc = tl.zeros((output_end - output_start,), dtype=tl.float32)
        
        # Loop over input dimension with block size BLOCK_K
        for k_start in range(0, hidden_size, BLOCK_K):
            k_end = min(k_start + BLOCK_K, hidden_size)
            k_mask = tl.arange(k_start, k_end) < hidden_size
            
            # Load input vector
            x_offset = batch_id * seq_len * hidden_size + seq_id * hidden_size + k_start
            x = tl.load(x_ptr + x_offset, mask=k_mask, other=0.0)
            
            # Load weight slice for current output chunk
            w_offset = output_start * hidden_size + k_start
            w = tl.load(w_ptr + w_offset, 
                       mask=tl.arange(0, (output_end - output_start) * (k_end - k_start)).reshape(output_end - output_start, k_end - k_start) < (output_end - output_start) * (k_end - k_start), 
                       other=0.0)
            w = w.reshape(output_end - output_start, k_end - k_start)
            
            # Load bias slice
            b_offset = output_start
            b = tl.load(b_ptr + b_offset, mask=tl.arange(output_start, output_end) < output_size, other=0.0)
            
            # Matrix multiplication: acc += x @ w.T
            acc += tl.dot(x, w.t())
        
        # Add bias and store result
        acc += b
        tl.store(y_ptr + y_offset + output_start, acc, mask=mask)

@torch.fx.wrap
def simple_linear_computation(in_3, weight, bias, batch_size, seq_len, hidden_size):
    # Get output dimensions
    output_size = weight.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, output_size), 
                        dtype=in_3.dtype, 
                        device=in_3.device)
    
    # Choose appropriate tile sizes
    BLOCK_M = 32   # Process 32 sequence positions per program
    BLOCK_N = 128  # Process 128 output features per block
    BLOCK_K = 32   # Process 32 input features per block
    
    # Calculate grid dimensions
    grid_size = (batch_size * seq_len + BLOCK_M - 1) // BLOCK_M
    
    # Launch kernel
    simple_linear_kernel[grid_size](
        in_3, weight, bias, output,
        batch_size, seq_len, hidden_size, output_size,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return output

def replacement_func():
    return simple_linear_computation