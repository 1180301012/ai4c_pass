import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the computation graph
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Matches the computation pattern: Conv2D -> LayerNorm -> ReLU
    - in_0: bias for Conv2D 
    - in_1: weight for Conv2D
    - in_2: bias for LayerNorm
    - in_3: weight for LayerNorm  
    - in_4: input tensor
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = torch.conv2d(in_4, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_5 = torch.nn.functional.layer_norm(tmp_4, (in_0.shape[0], 1, 1), tmp_3, tmp_2, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return (tmp_6,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

# Fused kernel using Triton
@triton.jit
def fused_conv_layernorm_relu_kernel(
    x_ptr,
    weight_ptr, 
    bias_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_height,
    in_width,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel: Conv2D (1x1) + LayerNorm + ReLU
    Optimized for small spatial dimensions (1x1) with varying channel sizes
    """
    # Get program IDs for grid-level parallelism
    pid_c = tl.program_id(0)  # output channel block
    pid_n = tl.program_id(1)  # batch element block
    
    # Calculate ranges for this program
    c_start = pid_c * BLOCK_SIZE_C
    n_start = pid_n * BLOCK_SIZE_N
    
    # Compute offsets for output channel dimensions
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < out_channels
    
    # Compute offsets for batch size dimensions  
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < batch_size
    
    # Load bias for this block of output channels
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)
    
    # Load LayerNorm weight for this block of output channels
    ln_weight = tl.load(ln_weight_ptr + c_offsets, mask=c_mask, other=1.0)
    
    # Load LayerNorm bias for this block of output channels
    ln_bias = tl.load(ln_bias_ptr + c_offsets, mask=c_mask, other=0.0)
    
    # Initialize output accumulator
    out = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=tl.float32)
    
    # Loop over input channels for convolution
    for k_base in range(0, in_channels, BLOCK_SIZE_C):
        k_offsets = k_base + tl.arange(0, BLOCK_SIZE_C)
        k_mask = k_offsets < in_channels
        
        # Load convolution weight for this input channel block and output channel block
        conv_weight = tl.load(
            weight_ptr + c_offsets[:, None] * in_channels + k_offsets[None, :],
            mask=c_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Load input data for all batch elements and this input channel block
        input_block = tl.load(
            x_ptr + n_offsets[:, None] * in_channels * in_height * in_width + 
            k_offsets[None, :] * in_height * in_width,
            mask=n_mask[:, None] & k_mask[None, :] & (in_height > 0) & (in_width > 0),
            other=0.0
        )
        
        # Convolution: multiply and accumulate
        out += input_block[:, :, None] * conv_weight[None, None, :]
    
    # Add bias
    out += bias[None, :]
    
    # LayerNorm: normalize and apply weights/biases
    # For 1x1 spatial, compute mean and variance across batch only
    if BLOCK_SIZE_N > 1:
        mean = tl.sum(out, axis=0) / BLOCK_SIZE_N
        var = tl.sum((out - mean[None, :]) ** 2, axis=0) / BLOCK_SIZE_N
    else:
        mean = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
        var = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    
    # Apply LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    eps = 1e-05
    normalized = (out - mean[None, :]) / tl.sqrt(var[None, :] + eps)
    normalized = normalized * ln_weight[None, :] + ln_bias[None, :]
    
    # Apply ReLU activation
    out = tl.maximum(normalized, 0.0)
    
    # Store results back to memory
    out_flat = out.reshape(-1)
    offsets = n_offsets[:, None] * out_channels + c_offsets[None, :]
    offsets_flat = offsets.reshape(-1)
    mask_flat = (n_mask[:, None] & c_mask[None, :]).reshape(-1)
    
    tl.store(out_ptr + offsets_flat, out_flat, mask=mask_flat)

# Kernel wrapper function
@torch.fx.wrap
def fused_conv_layernorm_relu(in_0, in_1, in_2, in_3, in_4):
    """
    Wrapper function to launch the fused kernel
    """
    # Get tensor shapes
    batch_size, in_channels, in_height, in_width = in_4.shape
    out_channels = in_0.shape[0]
    
    # Create output tensor
    out = torch.empty((batch_size, out_channels, in_height, in_width), 
                     dtype=in_4.dtype, device=in_4.device)
    
    # Choose block sizes based on tensor dimensions for optimal performance
    BLOCK_SIZE_C = min(64, out_channels)
    if batch_size <= 32:
        BLOCK_SIZE_N = min(32, batch_size)
    else:
        BLOCK_SIZE_N = 64
    
    # Calculate grid size
    grid_c = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_n = (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_conv_layernorm_relu_kernel[(grid_c, grid_n)](
        in_4,
        in_1,
        in_0, 
        in_3,
        in_2,
        out,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function - returns the fused kernel function
def replacement_func():
    return fused_conv_layernorm_relu