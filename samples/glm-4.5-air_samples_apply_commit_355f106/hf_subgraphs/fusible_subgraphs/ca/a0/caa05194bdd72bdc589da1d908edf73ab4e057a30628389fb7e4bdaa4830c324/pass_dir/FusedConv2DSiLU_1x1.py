import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    add_kernel[(num_programs,)](
        x_ptr=x, y_ptr=y, out_ptr=out, 
        n_elements=N, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return triton_add



@triton.jit
def fused_conv2d_silu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_height,
    in_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Program ID for parallel execution
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute memory bounds
    m_mask = pid_m < (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_mask = pid_n < (in_height * in_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    if not (m_mask and n_mask):
        return
    
    # Global offsets
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    b_offset = pid_b
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels (K dimension)
    for k in range(0, in_channels, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, in_channels)
        k_mask = k < in_channels
        
        # Load input tile: [batch_size, in_channels, in_height, in_width]
        # Reshape input to [batch_size, in_height*in_width, in_channels]
        input_idx = b_offset * in_channels * in_height * in_width + \
                   k * in_height * in_width + n_offset
        input_tile = tl.load(input_ptr + input_idx, mask=k_mask and tl.arange(0, BLOCK_SIZE_K)[:, None] < (k_end - k) * BLOCK_SIZE_N, 
                           other=0.0).reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)
        
        # Load weight tile: [out_channels, in_channels, 1, 1] -> [out_channels, in_channels]
        weight_idx = m_offset * in_channels + k
        weight_tile = tl.load(weight_ptr + weight_idx, mask=k_mask and tl.arange(0, BLOCK_SIZE_M)[:, None] < (k_end - k), 
                            other=0.0).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
        
        # Matrix multiply for this block
        acc += tl.dot(weight_tile, input_tile)
    
    # Load bias and add
    bias_idx = m_offset
    bias_tile = tl.load(bias_ptr + bias_idx, mask=tl.arange(0, BLOCK_SIZE_M) < (out_channels - m_offset), other=0.0).reshape(BLOCK_SIZE_M, 1)
    acc += bias_tile
    
    # Apply SiLU: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-acc))
    silu_out = acc * sigmoid_x
    
    # Store output
    output_idx = b_offset * out_channels * in_height * in_width + \
                m_offset * in_height * in_width + n_offset
    tl.store(output_ptr + output_idx, silu_out, mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < (out_channels - m_offset)[:, None] * \
             tl.arange(0, BLOCK_SIZE_N)[None, :] < (in_height * in_width - n_offset))

@torch.fx.wrap
def fused_conv2d_silu(input_tensor, weight_tensor, bias_tensor):
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, _, _ = weight_tensor.shape
    
    BLOCK_SIZE_M = 32  # Output channels
    BLOCK_SIZE_N = 64  # Spatial positions (H*W)
    BLOCK_SIZE_K = 32  # Input channels
    
    # Calculate grid size
    grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (in_height * in_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_b = batch_size
    
    output = torch.empty((batch_size, out_channels, in_height, in_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    fused_conv2d_silu_kernel[(grid_m, grid_n, grid_b)](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return fused_conv2d_silu