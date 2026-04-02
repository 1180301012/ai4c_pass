import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match: batch_norm + add + relu + spatial_mean
    in_0: running_mean
    in_1: running_var  
    in_2: bias
    in_3: weight
    in_4: input tensor 1
    in_5: input tensor 2 (added after batch norm)
    """
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

# Optimized kernel for batch_norm + add + relu fusion
@triton.jit
def fused_batchnorm_add_relu_kernel(
    x_ptr,                    # Input tensor 4 (main input)
    add_ptr,                  # Input tensor 5 (to be added)
    running_mean_ptr,         # Batch norm running mean
    running_var_ptr,          # Batch norm running variance  
    weight_ptr,               # Batch norm weight
    bias_ptr,                 # Batch norm bias
    out_ptr,                  # ReLU output (for tmp_6)
    mean_ptr,                 # Mean output (for tmp_7)
    n_channels,               # Number of channels (C)
    height,                   # Height (H)
    width,                    # Width (W)
    eps: tl.constexpr,        # Epsilon for batch norm
    momentum: tl.constexpr,   # Momentum for batch norm
    BLOCK_SIZE_M: tl.constexpr,  # Block size for channels
    BLOCK_SIZE_N: tl.constexpr   # Block size for spatial dimensions
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, n_channels)
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, height * width)
    
    # Loop through channels and spatial dimensions
    for m_off in range(m_start, m_end, 1):
        for n_off in range(n_start, n_end, 1):
            # Compute offsets
            x_offset = m_off * height * width + n_off
            add_offset = m_off * height * width + n_off
            
            # Load batch norm parameters (they're constants based on weight_meta)
            weight_val = tl.load(weight_ptr + m_off)
            bias_val = tl.load(bias_ptr + m_off)
            running_mean_val = tl.load(running_mean_ptr + m_off)
            running_var_val = tl.load(running_var_ptr + m_off)
            
            # Load input values
            x_val = tl.load(x_ptr + x_offset)
            add_val = tl.load(add_ptr + add_offset)
            
            # Apply batch normalization
            denom = tl.sqrt(running_var_val + eps)
            batch_norm_val = (x_val - running_mean_val) / denom * weight_val + bias_val
            
            # Add the second input
            add_result = batch_norm_val + add_val
            
            # Apply ReLU
            relu_val = tl.maximum(add_result, 0.0)
            
            # Store ReLU output
            tl.store(out_ptr + x_offset, relu_val)
    
    # Special kernel for mean reduction (separate for better performance)
    if pid_n == 0 and pid_m == 0:
        # Compute mean over spatial dimensions (2,3) for each channel
        for c in range(n_channels):
            total = 0.0
            for h in range(height):
                for w in range(width):
                    offset = c * height * width + h * width + w
                    val = tl.load(out_ptr + offset)
                    total += val
            
            mean_val = total / (height * width)
            tl.store(mean_ptr + c, mean_val)

@torch.fx.wrap
def fused_batchnorm_add_relu_forward(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Fused forward pass: batch_norm + add + relu + spatial_mean
    """
    # Get tensor shapes
    B, C, H, W = in_4.shape
    assert in_5.shape == (B, C, H, W), "Input tensors must have compatible shapes"
    
    # Create output tensors
    relu_out = torch.empty_like(in_4)
    mean_out = torch.empty(B, C, 1, 1, device=in_4.device, dtype=in_4.dtype)
    
    # Extract batch norm parameters for this batch
    running_mean = in_0  # Shape: [C]
    running_var = in_1   # Shape: [C] 
    weight = in_3        # Shape: [C]
    bias = in_2          # Shape: [C]
    
    # Launch kernel for each batch
    for b in range(B):
        # Compute offsets for this batch
        x_ptr = in_4[b].flatten().contiguous()
        add_ptr = in_5[b].flatten().contiguous()
        relu_out_ptr = relu_out[b].flatten().contiguous()
        mean_out_ptr = mean_out[b].flatten().contiguous()
        
        # Launch grid
        def grid(meta):
            return (
                triton.cdiv(C, meta['BLOCK_SIZE_M']),
                triton.cdiv(H * W, meta['BLOCK_SIZE_N'])
            )
        
        # Launch the fused kernel
        fused_batchnorm_add_relu_kernel[grid](
            x_ptr=x_ptr,
            add_ptr=add_ptr,
            running_mean_ptr=running_mean,
            running_var_ptr=running_var,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=relu_out_ptr,
            mean_ptr=mean_out_ptr,
            n_channels=C,
            height=H,
            width=W,
            eps=1e-05,
            momentum=0.1,
            BLOCK_SIZE_M=32,      # Optimal for channel parallelism
            BLOCK_SIZE_N=1024     # Optimal for spatial parallelism
        )
    
    return (relu_out, mean_out)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_batchnorm_add_relu_forward