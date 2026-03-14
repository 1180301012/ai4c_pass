import torch
import triton
import triton.language as tl
import math

def pattern(x, y):
    # Create an extremely simple pattern to test if matching works at all
    a = x + y
    b = a * 2
    return b, a

def replacement_args(x, y):
    return (x, y)

@triton.jit
def optimized_conv_slice_kernel(
    input_ptr,  # [in_channels, H_in, W_in]
    weight_ptr,  # [out_channels, in_channels, kH, kW]
    output_ptr,  # [num_slice_output_channels, H_out, W_out]
    output_full_ptr,  # [full_out_channels, H_out, W_out]
    in_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    num_slice_outputs,  # Number of output channels to compute (e.g., 1024)
    full_out_channels,  # Total output channels that would be computed normally
    stride_h,
    stride_w,
    kernel_h,
    kernel_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute ranges
    m_range = tl.device_constexpr(lambda: [BLOCK_SIZE_M])()[0]
    n_range = tl.device_constexpr(lambda: [BLOCK_SIZE_N])[0]
    k_range = tl.device_constexpr(lambda: [BLOCK_SIZE_K])[0]
    
    # Offset for output channels
    m_offsets = pid_m * m_range + tl.arange(0, m_range)
    m_mask = m_offsets < num_slice_outputs
    
    # Offset for output spatial positions
    n_offsets = pid_n * n_range + tl.arange(0, n_range)
    n_mask = n_offsets < out_height * out_width
    
    # Offset for input channels
    k_offsets = pid_k * k_range + tl.arange(0, k_range)
    k_mask = k_offsets < in_channels
    
    # Initialize accumulator
    acc = tl.zeros((m_range, n_range), dtype=tl.float32)
    
    # Load input and compute matrix multiplication
    for k in range(0, in_channels, BLOCK_SIZE_K):
        k_block_end = min(k + BLOCK_SIZE_K, in_channels)
        k_local = tl.arange(k, k_block_end)
        
        # Load input block
        input_block = tl.load(
            input_ptr + k_local.reshape(1, 1, -1) + 
            tl.zeros((m_range, n_range, k_block_end - k), dtype=tl.int32),
            mask=k_local.reshape(1, 1, -1) < in_channels,
            other=0.0
        )
        
        # Reshape input for matrix multiplication
        input_reshaped = input_block.reshape(m_range, n_range, -1)
        
        # For each output channel in current block
        for m in range(pid_m * m_range, min((pid_m + 1) * m_range, num_slice_outputs)):
            # Load weight for this output channel
            weight_block = tl.load(
                weight_ptr + m.reshape(1, 1, -1) + 
                k_local.reshape(1, -1, 1),
                mask=(m < num_slice_outputs) & (k_local < in_channels),
                other=0.0
            )
            
            # Reshape weight for matrix multiplication
            weight_reshaped = weight_block.reshape(1, k_block_end - k, -1)
            
            # Compute product and accumulate
            prod = tl.sum(input_reshaped * weight_reshaped, dim=2)
            acc += prod
    
    # Compute output position
    h_idx = n_offsets // out_width
    w_idx = n_offsets % out_width
    out_idx = m_offsets.reshape(-1, 1) + h_idx.reshape(1, -1) * num_slice_outputs + w_idx.reshape(1, -1) * num_slice_outputs * out_height
    
    # Store sliced output
    tl.store(
        output_ptr + out_idx,
        acc,
        mask=m_mask.reshape(-1, 1) & n_mask.reshape(1, -1)
    )

@torch.fx.wrap
def simple_kernel(x, y):
    # Very simple kernel for testing pattern matching
    a = x + y
    b = a * 2
    return b, a

def replacement_func():
    return simple_kernel