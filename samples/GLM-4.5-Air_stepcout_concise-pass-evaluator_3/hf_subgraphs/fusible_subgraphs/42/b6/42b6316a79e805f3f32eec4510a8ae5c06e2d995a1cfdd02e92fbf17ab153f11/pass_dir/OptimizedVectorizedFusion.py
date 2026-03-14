import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Match Conv2D + Hardswish sequence"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(tmp_2, True)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def vectorized_conv_hardswish_kernel(
    input_ptr,  # [B, C_in, H, W]
    weight_ptr, # [C_out, C_in, K, K]
    bias_ptr,   # [C_out]
    output_ptr, # [B, C_out, H, W]
    B, C_in, C_out, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program IDs
    pid_m = tl.program_id(0)  # Output channel group
    pid_n = tl.program_id(1)  # Batch group
    
    # Offsets within this program group
    m_offsets = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_offsets = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks
    m_mask = m_offsets < C_out
    n_mask = n_offsets < B
    
    # Process all combinations of (channel, batch) in this program
    for m_idx in range(BLOCK_SIZE):
        m = m_offsets[m_idx]
        if not m_mask[m_idx]:
            continue
            
        # Load bias for this output channel
        bias_val = tl.load(bias_ptr + m)
        
        # Load weight slice for this output channel
        k_offsets = tl.arange(0, C_in)
        weight_slice = tl.load(weight_ptr + m * C_in + k_offsets)
        
        for n_idx in range(BLOCK_SIZE):
            n = n_offsets[n_idx]
            if not n_mask[n_idx]:
                continue
            
            # Load input slice for this batch item
            input_slice = tl.load(input_ptr + n * C_in + k_offsets)
            
            # Compute convolution using vectorized dot product
            conv_result = tl.sum(weight_slice * input_slice) + bias_val
            
            # Apply vectorized hardswish
            relu6_val = tl.maximum(tl.minimum(conv_result + 3.0, 6.0), 0.0)
            hardswish_result = conv_result * relu6_val / 6.0
            
            # Store result
            output_offset = n * C_out + m
            tl.store(output_ptr + output_offset, hardswish_result)

@torch.fx.wrap
def vectorized_fused_conv_hardswish(in_0, in_1, in_2):
    # Get input shapes
    B, C_in, H, W = in_2.shape
    C_out = in_1.shape[0]  # Weight tensor shape: [C_out, C_in, K, K]
    
    # Create output tensor
    output_shape = (B, C_out, H, W)
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Optimized block size for better GPU utilization
    if C_out * B < 1024:
        BLOCK_SIZE = 16
    else:
        BLOCK_SIZE = 32
    
    # Calculate number of programs needed
    num_programs_m = (C_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_programs_n = (B + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    vectorized_conv_hardswish_kernel[(num_programs_m, num_programs_n)](
        in_2, in_1, in_0, output,
        B, C_in, C_out, H, W,
        BLOCK_SIZE
    )
    
    # Flatten the output to match original behavior
    return output.flatten(1, -1)

def replacement_func():
    return vectorized_fused_conv_hardswish