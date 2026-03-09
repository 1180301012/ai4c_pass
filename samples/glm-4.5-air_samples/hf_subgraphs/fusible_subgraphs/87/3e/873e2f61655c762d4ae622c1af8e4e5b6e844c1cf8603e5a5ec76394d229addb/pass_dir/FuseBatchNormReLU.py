import torch
import triton
import triton.language as tl

# Pattern matching function for BatchNorm + ReLU fusion
def pattern(tmp_8, tmp_1, tmp_2, tmp_4, tmp_3):
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
    tmp_10 = torch.nn.functional.relu(tmp_9, inplace=True)
    return tmp_10  # Only return the observable value that's used downstream (the final result)

# Argument extraction function
def replacement_args(tmp_8, tmp_1, tmp_2, tmp_4, tmp_3):
    return (tmp_8, tmp_1, tmp_2, tmp_4, tmp_3)

# Optimized kernel for fused BatchNorm + ReLU
@triton.jit
def fused_batchnorm_relu_kernel(
    x_ptr,          # tmp_8: input tensor [B, C, H, W]
    running_mean_ptr,  # tmp_1: running mean [C]
    running_var_ptr,   # tmp_2: running variance [C]
    weight_ptr,        # tmp_4: weight [C]
    bias_ptr,          # tmp_3: bias [C]
    out_ptr,           # output temporary [B, C, H, W]
    B, C, H, W,        # Tensor dimensions
    BLOCK_SIZE_M: tl.constexpr,  # Block size for channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for spatial dimensions
):
    # Get program IDs
    pid_m = tl.program_id(0)  # Channel blocks
    pid_n = tl.program_id(1)  # Spatial blocks
    
    # Calculate ranges
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Create offsets for channels
    channel_offsets = m_offset + tl.arange(0, BLOCK_SIZE_M)
    channel_mask = channel_offsets < C
    
    # Load parameters for this channel block
    means = tl.load(running_mean_ptr + channel_offsets, mask=channel_mask)
    vars = tl.load(running_var_ptr + channel_offsets, mask=channel_mask)
    weights = tl.load(weight_ptr + channel_offsets, mask=channel_mask)
    biases = tl.load(bias_ptr + channel_offsets, mask=channel_mask)
    
    # Compute scale and bias: y = (x - mean) / sqrt(var + eps) * weight + bias
    eps = 1e-05
    sqrt_vars = tl.sqrt(vars + eps)
    inv_vars = 1.0 / sqrt_vars
    
    # Precompute scale and bias
    scale = weights * inv_vars
    bias = biases - means * scale
    
    # Process a block of spatial elements
    n_end = min(n_offset + BLOCK_SIZE_N, H * W)
    for spatial_idx in range(n_offset, n_end):
        # Load input data for all channels in this spatial position
        spatial_offset = spatial_idx * C
        x = tl.load(x_ptr + spatial_offset + channel_offsets, mask=channel_mask)
        
        # Apply batch normalization
        norm = (x - means) * scale + bias
        
        # Apply ReLU
        relu_out = tl.maximum(norm, 0.0)
        
        # Store result
        tl.store(out_ptr + spatial_offset + channel_offsets, relu_out, mask=channel_mask)

@torch.fx.wrap
def fused_batchnorm_relu(tmp_8, tmp_1, tmp_2, tmp_4, tmp_3):
    # Get tensor shapes
    B, C, H, W = tmp_8.shape
    
    # Output shape should be [B, C, H, W]
    out = torch.empty((B, C, H, W), dtype=torch.float32, device=tmp_8.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 128  # Number of channels to process per program
    BLOCK_SIZE_N = 256  # Number of spatial elements to process per program
    
    # Calculate grid size
    num_programs_m = (C + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (H * W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_batchnorm_relu_kernel[(num_programs_m, num_programs_n)](
        x_ptr=tmp_8,
        running_mean_ptr=tmp_1,
        running_var_ptr=tmp_2,
        weight_ptr=tmp_4,
        bias_ptr=tmp_3,
        out_ptr=out,
        B=B, C=C, H=H, W=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_batchnorm_relu