import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly mirror the model.py operations
def pattern(in_0, in_1, in_2):
    """
    Pattern matches conv2d + flatten operations as seen in model.py
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized fused kernel - matrix multiplication approach for 1x1 conv
@triton.jit
def fused_conv2d_flatten_kernel_gemm(
    input_ptr,      # in_2: [N, C_in, H, W]  -> treated as [N, C_in, H*W]
    weight_ptr,     # in_1: [C_out, C_in, 1, 1] -> treated as [C_out, C_in]
    bias_ptr,       # in_0: [C_out]
    output_ptr,     # flattened output: [N, C_out, H*W]
    N,              # Batch size
    C_in,           # Input channels (160)
    C_out,          # Output channels (17)
    H,              # Height (64)
    W,              # Width (48)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program computes one output element (batch, out_channel, spatial)
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    spatial_idx = tl.program_id(2)
    
    # Check bounds
    if batch_idx >= N:
        return
    if out_channel_idx >= C_out:
        return
    if spatial_idx >= H * W:
        return
    
    # Extract spatial coordinates
    h = spatial_idx // W
    w = spatial_idx % W
    
    # Calculate output index
    output_idx = batch_idx * C_out * H * W + out_channel_idx * H * W + spatial_idx
    
    # Load bias
    bias_val = tl.load(bias_ptr + out_channel_idx)
    acc = bias_val
    
    # Matrix multiplication: output = input @ weight.T + bias
    # Input: [1, C_in], Weight: [C_out, C_in], Output: [1, 1]
    for k in range(C_in):
        # Load input element at [batch_idx, k, h, w]
        input_idx = batch_idx * C_in * H * W + k * H * W + h * W + w
        input_val = tl.load(input_ptr + input_idx)
        
        # Load weight at [out_channel_idx, k, 0, 0]
        weight_idx = out_channel_idx * C_in * 1 * 1 + k * 1 * 1 + 0 * 1 + 0
        weight_val = tl.load(weight_ptr + weight_idx)
        
        acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + output_idx, acc)

@torch.fx.wrap  
def fused_conv2d_flatten_call(in_0, in_1, in_2):
    """
    Wrapper function that uses a GEMM-style approach for the fused operation
    """
    N, C_in, H, W = in_2.shape
    C_out = in_0.shape[0]
    
    # Flatten spatial dimensions before GEMM
    # Note: conv2d with 1x1 kernel followed by flatten(dim=2) is equivalent to:
    # Reshape input to [N*H*W, C_in], reshape output to [N*H*W, C_out]
    # and perform matrix multiplication: output = input @ weight.T + bias
    
    total_spatial = H * W
    
    # Calculate 3D grid dimensions for GEMM approach
    # Each program computes one output element (batch, out_channel, spatial)
    grid_size_0 = N
    grid_size_1 = C_out
    grid_size_2 = total_spatial
    
    # Allocate output tensor with flattened spatial dimension
    output = torch.empty((N, C_out, total_spatial), dtype=in_2.dtype, device=in_2.device)
    
    # For this simple GEMM approach, we don't need blocking parameters
    # This creates one program per output element
    total_output_elements = N * C_out * total_spatial
    
    # If there are too many programs, we might need blocking, but let's try simple first
    if total_output_elements > 1000000:  # If more than 1M elements
        # Use blocking to reduce program count
        BLOCK_N = 8  # Process 8 output channels per program
        BLOCK_SPATIAL = 64  # Process 64 spatial locations per program
        
        grid_size_0 = N
        grid_size_1 = (C_out + BLOCK_N - 1) // BLOCK_N
        grid_size_2 = (total_spatial + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL
        
        fused_conv2d_flatten_kernel_gemm[(grid_size_0, grid_size_1, grid_size_2)](
            input_ptr=in_2,
            weight_ptr=in_1,
            bias_ptr=in_0,
            output_ptr=output,
            N=N,
            C_in=C_in,
            C_out=C_out,
            H=H,
            W=W,
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_SPATIAL,
        )
    else:
        # Simple launch for smaller workloads
        fused_conv2d_flatten_kernel_gemm[(grid_size_0, grid_size_1, grid_size_2)](
            input_ptr=in_2,
            weight_ptr=in_1,
            bias_ptr=in_0,
            output_ptr=output,
            N=N,
            C_in=C_in,
            C_out=C_out,
            H=H,
            W=W,
            BLOCK_SIZE_M=1,
            BLOCK_SIZE_N=1,
            BLOCK_SIZE_K=1,
        )
    
    return output

# Replacement function (no arguments, returns callable)
def replacement_func():
    return fused_conv2d_flatten_call