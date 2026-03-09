import torch
import triton
import triton.language as tl

def pattern(args):
    input, weight, bias = args
    conv = torch.conv2d(input, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    gelu = torch.nn.functional.gelu(conv)
    return gelu

def replacement_args(args):
    return args

@triton.jit
def fused_conv1x1_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Determine range of rows (output channels) each program handles
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < out_channels
    m_offsets = m_offsets[m_mask]
    
    if len(m_offsets) == 0:
        return
    
    # Determine range of spatial locations each program handles
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < in_height * in_width
    n_offsets = n_offsets[n_mask]
    n_coords_h = n_offsets // in_width
    n_coords_w = n_offsets % in_width
    
    bias_vector = tl.load(bias_ptr + m_offsets, mask=m_mask)
    
    # Initialize accumulator
    accumulator = tl.zeros((len(m_offsets), len(n_offsets)), dtype=tl.float32)
    
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Main computation loop
    for k_base in range(0, in_channels, BLOCK_SIZE_K):
        # Create masks for bounds checking
        k_mask = k_base + k_offsets < in_channels
        k_offsets_valid = k_base + k_offsets[k_mask]
        
        # Load input slice
        input_ptrs = input_ptr + (n_coords_h[:, None] * in_width * in_channels + 
                                  n_coords_w[:, None] * in_channels + 
                                  k_offsets_valid[None, :])
        input_slice = tl.load(input_ptrs, mask=k_offsets_valid[None, :] >= 0, other=0.0)
        
        # Load weight slice
        weight_ptrs = weight_ptr + (m_offsets[:, None] * in_channels + k_offsets_valid[None, :])
        weight_slice = tl.load(weight_ptrs, mask=k_offsets_valid[None, :] >= 0, other=0.0)
        
        # Matrix multiplication
        accumulator += input_slice * weight_slice[None, :]
    
    # Apply GELU activation
    gelu_output = 0.5 * accumulator * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.pi) * (accumulator + 0.044715 * accumulator * accumulator * accumulator)))
    
    # Add bias
    final_output = gelu_output + bias_vector[:, None]
    
    # Write result to output
    output_ptrs = output_ptr + (m_offsets[:, None] * in_height * in_width + n_offsets[None, :])
    tl.store(output_ptrs, final_output, mask=m_offsets[:, None] < out_channels)

@torch.fx.wrap
def fused_conv1x1_gelu(input, weight, bias):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, _ = weight.shape
    
    # For 1x1 conv with padding=0: output size is same as input for spatial dims
    out_height = in_height
    out_width = in_width
    
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)
    
    # Block sizes optimized for GPU
    BLOCK_SIZE_M = 64   # Output channels per program
    BLOCK_SIZE_N = 1024  # Spatial locations per program
    BLOCK_SIZE_K = 32   # Input channels per loop
    
    # Grid calculation
    num_programs_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (in_height * in_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_conv1x1_gelu_kernel[(num_programs_m, num_programs_n)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return fused_conv1x1_gelu