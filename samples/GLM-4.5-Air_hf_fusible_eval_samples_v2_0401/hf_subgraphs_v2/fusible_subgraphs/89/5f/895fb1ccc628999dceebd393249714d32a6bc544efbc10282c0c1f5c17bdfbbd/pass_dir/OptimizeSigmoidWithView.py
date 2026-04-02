import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation to be replaced
def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Optimized Triton kernel - complete fused computation with better memory access
@triton.jit
def fused_computation_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs for 2D grid (channels X spatial)
    pid_m = tl.program_id(0)  # channel block
    pid_n = tl.program_id(1)  # spatial block
    
    # Compute offsets for this thread block
    channel_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    spatial_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    channel_mask = channel_offset < C
    spatial_mask = spatial_offset < (H * W)
    
    # Precompute sigmoid weights for this channel block
    sigmoid_weights = tl.load(in_0_ptr + channel_offset, mask=channel_mask, other=0.0)
    
    # Compute linear offsets for all combinations of channels and spatial positions
    # flattened_idx = c * (H * W) + spatial_idx
    flattened_offsets = channel_offset[:, None] * (H * W) + spatial_offset[None, :]
    combined_mask = channel_mask[:, None] & spatial_mask[None, :]
    
    # Load input data for all combinations efficiently
    in_1_vals = tl.load(in_1_ptr + flattened_offsets, mask=combined_mask, other=0.0)
    
    # Apply fused operation: in_1 * (1 + sigmoid_weight) with broadcasting
    # sigmoid_weights has shape [BLOCK_SIZE_M], in_1_vals has shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
    fused_vals = in_1_vals * (1.0 + sigmoid_weights[:, None])
    
    # Apply ReLU activation
    relu_vals = tl.maximum(fused_vals, 0.0)
    
    # Store results back to memory
    tl.store(out_ptr + flattened_offsets, relu_vals, mask=combined_mask)

# Kernel wrapper for complete fused computation
@torch.fx.wrap
def fused_computation(in_0, in_1):
    # Get tensor shapes
    N, C, H, W = in_1.shape
    
    # Set up Triton kernel parameters - use 2D tiling for better memory access
    BLOCK_SIZE_M = 64   # Block size for channels (channels dimension)
    BLOCK_SIZE_N = 1024 # Block size for spatial dimensions (H*W)
    
    # Calculate grid size for 2D parallelization
    num_channel_blocks = (C + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_spatial_blocks = (H * W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Launch Triton kernel with 2D grid
    fused_computation_kernel[(num_channel_blocks, num_spatial_blocks)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function (returns the fused computation function)
def replacement_func():
    return fused_computation