import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(tmp_2, 2)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv2d_flatten_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    n_input_features,
    n_output_features,
    n_elements,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs: batch, output channels, spatial positions
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Compute offsets with proper boundaries
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    m_mask = m_offsets < n_output_features
    n_mask = n_offsets < n_elements
    
    # Flatten masks for storage
    m_mask_flat = m_mask[:, None] & n_mask[None, :]
    
    # Calculate global memory offset for this batch
    batch_offset = pid_b * n_input_features * n_elements
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Optimized loop with better memory access pattern
    for k in range(n_input_features):
        # Load weights: [output_features, input_features]
        weight_ptr_base = weight_ptr + m_offsets[:, None] * n_input_features + k
        weight = tl.load(weight_ptr_base, mask=m_mask[:, None], other=0.0)
        
        # Load input for this batch and feature: [spatial_elements]
        input_ptr_base = input_ptr + batch_offset + k * n_elements + n_offsets[None, :]
        input_val = tl.load(input_ptr_base, mask=n_mask[None, :], other=0.0)
        
        # Vectorized matrix multiplication
        accumulator += input_val * weight
    
    # Add bias with broadcasting
    bias_ptr_base = bias_ptr + m_offsets
    bias_val = tl.load(bias_ptr_base, mask=m_mask, other=0.0)
    bias_val = bias_val[:, None]
    accumulator += bias_val
    
    # Optimized store with fused computation
    output_ptr_base = output_ptr + (pid_b * n_output_features * n_elements + 
                                   m_offsets[:, None] * n_elements + n_offsets[None, :])
    tl.store(output_ptr_base, accumulator, mask=m_mask_flat)

@torch.fx.wrap
def fused_conv2d_flatten(in_0, in_1, in_2):
    # Get input dimensions
    batch_size, input_channels, height, width = in_2.shape
    output_channels = in_0.shape[0]
    spatial_total = height * width
    
    # Calculate output shape: [batch_size, output_channels, spatial_total]
    output = torch.empty((batch_size, output_channels, spatial_total), 
                        dtype=torch.float32, device=in_2.device)
    
    # 1x1 convolutions with flattening can be optimized using efficient matmul
    # For small spatial_total (3072) and medium input_channels (160), 
    # we can use optimized matrix multiplication
    
    if spatial_total <= 4096:  # Use optimized matmul approach for moderate spatial sizes
        # Reshape for matrix multiplication: [batch_size, input_channels, spatial_total]
        input_3d = in_2.reshape(batch_size, input_channels, spatial_total)
        
        # Optimized block sizes for this workload
        if batch_size >= 128:
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 256
        elif batch_size >= 32:
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 128
        else:
            # Small batch size - use block sizes that maximize occupancy
            BLOCK_SIZE_M = 16
            BLOCK_SIZE_N = 512
        
        # Calculate grid dimensions
        num_b_programs = batch_size
        num_m_programs = (output_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        num_n_programs = (spatial_total + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        grid = (num_b_programs, num_m_programs, num_n_programs)
        
        # Launch kernel
        fused_conv2d_flatten_kernel[grid](
            input_ptr=input_3d,
            weight_ptr=in_1.reshape(output_channels, input_channels),
            bias_ptr=in_0,
            output_ptr=output,
            batch_size=batch_size,
            n_input_features=input_channels,
            n_output_features=output_channels,
            n_elements=spatial_total,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    else:
        # For larger spatial sizes, fall back to direct PyTorch operations
        # which are already optimized for these cases
        conv_out = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
        output = torch.flatten(conv_out, 2)
    
    return output

def replacement_func():
    return fused_conv2d_flatten