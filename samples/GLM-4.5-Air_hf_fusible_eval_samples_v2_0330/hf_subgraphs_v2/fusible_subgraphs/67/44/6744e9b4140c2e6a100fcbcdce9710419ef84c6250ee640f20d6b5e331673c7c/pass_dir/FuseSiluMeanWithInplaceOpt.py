import torch
import triton
import triton.language as tl

def pattern(in_1):
    """
    Pattern matching for SILU activation followed by mean reduction.
    This is the most expensive operation happening on large tensors [1, 384, H, W].
    """
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    return tmp_0, tmp_1

def replacement_args(in_1):
    # Extract the input tensor
    return (in_1,)

@triton.jit
def fused_silu_mean_kernel(
    input_ptr,
    output_silu_ptr,
    output_mean_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel that fuses SILU activation with mean reduction.
    This reduces memory bandwidth by computing both operations in one pass.
    """
    # Program identifiers
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Calculate bounds
    m_mask = m_offset < n_channels
    n_mask = n_offset < height * width
    
    # Load input data - store in shared memory for better performance
    input_ptrs = input_ptr + m_offset * height * width + n_offset
    input_data = tl.load(input_ptrs, mask=n_mask, other=0.0)
    
    # Compute SILU activation in-place: silu(x) = x * sigmoid(x)
    sigmoid_data = 1.0 / (1.0 + tl.exp(-input_data))
    silu_data = input_data * sigmoid_data
    
    # Store SILU output
    silu_ptrs = output_silu_ptr + m_offset * height * width + n_offset
    tl.store(silu_ptrs, silu_data, mask=n_mask)
    
    # Compute mean for this block
    if m_mask:
        # Reduce across spatial dimensions
        block_sum = tl.sum(silu_data)
        block_mean = block_sum / (height * width)
        
        # Store mean output (channel-wise)
        mean_ptrs = output_mean_ptr + pid_m
        tl.store(mean_ptrs, block_mean)

@torch.fx.wrap
def fused_silu_mean_optimized(input_tensor):
    """
    Wrapper function that launches the fused SILU + mean kernel.
    Input shape: [1, C, H, W]
    Output shapes: silu_output [1, C, H, W], mean_output [1, C, 1, 1]
    """
    # Unpack tensor dimensions
    batch_size, channels, height, width = input_tensor.shape
    assert batch_size == 1, "Only batch size 1 is supported"
    
    # Create output tensors
    silu_output = torch.zeros_like(input_tensor)
    mean_output = torch.zeros(1, channels, 1, 1, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Configure block sizes for optimal GPU occupancy
    BLOCK_SIZE_M = 64  # Number of channels per block
    BLOCK_SIZE_N = 1024  # Number of spatial elements per block
    
    # Calculate grid dimensions
    grid_m = (channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (height * width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch the fused kernel
    fused_silu_mean_kernel[(grid_m, grid_n)](
        input_ptr=input_tensor,
        output_silu_ptr=silu_output,
        output_mean_ptr=mean_output,
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return silu_output, mean_output.squeeze(-1).squeeze(-1).unsqueeze(-1)

def replacement_func():
    """Returns the optimized fused function"""
    return fused_silu_mean_optimized