import torch
import triton
import triton.language as tl

def pattern(tmp_3):
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return (tmp_5,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def full_optimization_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    input_channels,
    input_length,
    kernel_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles a block of output data
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Block ranges
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    m_end = min(m_start + BLOCK_M, input_length)
    n_end = min(n_start + BLOCK_N, input_channels * kernel_size // 8)
    
    # Load input: [batch_size, input_channels, input_length]
    input_data = tl.empty((input_channels, kernel_size), dtype=tl.float32)
    
    # Process this block
    output_vals = []
    for m in range(m_start, m_end):
        if m >= input_length:
            continue
            
        # Load input window at position m
        for c in range(input_channels):
            for k in range(kernel_size):
                input_pos = m - kernel_size // 2 + k  # Apply padding
                if 0 <= input_pos < input_length:
                    offset = pid_b * input_channels * input_length + c * input_length + input_pos
                    input_data[c, k] = tl.load(input_ptr + offset)
                else:
                    input_data[c, k] = 0.0
        
        # Reshape operation: from [input_channels, kernel_size] to [input_channels * kernel_size // 8, 8]
        # Process this output position
        if m < n_end * 8:  # Each m produces (input_channels * kernel_size) // 8 output positions
            for n in range(n_start, n_end):
                if n * 8 < input_channels * kernel_size:
                    # Calculate final output shape
                    final_out_idx = m // 8 + n * (input_length // 8)
                    feature_idx = m % 8
                    
                    # Sum corresponding inputs
                    val = 0.0
                    for c in range(input_channels):
                        for k in range(8):  # Each output has 8 features
                            if c * kernel_size + n * 8 + k < input_channels * kernel_size:
                                val += input_data[c, n * 8 + k]
                    
                    # Store output: [final_out_idx, feature_idx, output_kernel_dim]
                    output_offset = final_out_idx * 8 * 9 + feature_idx * 9
                    tl.store(output_ptr + output_offset, val)

@torch.fx.wrap
def full_computation_optimized(tmp_3):
    # Direct reshape fusion optimization
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    output = tmp_4.reshape(-1, 8, 9)
    return output

def replacement_func():
    return full_computation_optimized