import torch
import triton
import triton.language as tl


def pattern(in_5, in_1, in_2, in_3, in_4):
    """
    Pattern to match: adaptive_avg_pool2d -> batch_norm -> relu
    
    This pattern matches:
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    
    IMPORTANT: Must use exact same argument order and types as in model.py
    - batch_norm(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05)
    - relu(input, inplace=True)
    """
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    return tmp_8


def replacement_args(in_5, in_1, in_2, in_3, in_4):
    """
    Extract arguments needed for the replacement kernel.
    
    in_5: input tensor (batch, channels, H, W)
    in_1: running_mean (channels,)
    in_2: running_var (channels,) 
    in_3: bias (channels,)
    in_4: weight (channels,)
    """
    return (in_5, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
    ],
    key=['num_channels'],
)
@triton.jit
def fused_adaptive_avg_pool2d_batch_norm_relu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    bias_ptr,
    weight_ptr,
    output_ptr,
    num_channels,
    batch_size,
    # Per-channel stats (precomputed)
    # Actually we compute on-the-fly for this pattern
    # since we need to do adaptive avg pool first
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Adaptive average pooling to (1, 1)
    2. Batch normalization (using running mean/var)
    3. ReLU activation
    
    This is equivalent to:
    tmp_6 = F.adaptive_avg_pool2d(input, (1,1))
    tmp_7 = F.batch_norm(tmp_6, running_mean, running_var, weight, bias, training=False, momentum, eps)
    tmp_8 = F.relu(tmp_7, inplace=True)
    """
    # Each program processes one channel for all batch elements
    # We need to compute avg pool first, which requires scanning spatial dimensions
    
    # Strategy: Each program handles one channel
    # We process all spatial positions for all batch elements for that channel
    # Then apply BN and ReLU
    
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # For the adaptive avg pool to (1,1), we need to average all spatial positions
    # Get the input shape info from pointers (we'll use BLOCK_SIZE for spatial)
    # Actually, let's process per-channel with a different approach
    
    # Grid: (num_channels,)
    if pid >= num_channels:
        return
    
    # Load batch normalization parameters for this channel
    mean = tl.load(running_mean_ptr + pid)
    var = tl.load(running_var_ptr + pid)
    bias = tl.load(bias_ptr + pid)
    weight = tl.load(weight_ptr + pid)
    
    # Compute normalized value: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = tl.rsqrt(var + eps)
    scale = weight * inv_std
    bias_term = bias - mean * weight * inv_std
    
    # Now we need to compute adaptive avg pool: average over all spatial dims
    # For input of shape (batch, channels, H, W), output is (batch, channels, 1, 1)
    # We process each batch element
    
    # The input stride info - we need to know H and W
    # We can't get that directly, so we assume the kernel is called with proper config
    # Let's compute the sum over spatial dimensions
    
    # Actually, we need H and W from the input
    # Let's restructure: we'll compute this differently
    
    # Since we're doing adaptive_avg_pool2d to (1,1), we need to average all spatial positions
    # This requires knowing H and W
    
    # For the output, we just need the final value per channel per batch
    # Let's process it in a loop over spatial positions
    
    # We'll rely on the caller to provide proper grid
    # But wait - the autotune key uses num_channels, so grid should be num_channels
    
    # Load the output pointer offset for this channel
    output_offset = pid
    
    # We'll compute the sum in a loop - but we don't know H*W
    # Let's have the kernel compute the sum of all elements for this channel
    # This requires knowing the total number of elements per channel
    
    # Actually, let's try a simpler approach: 
    # We'll process the data in a way that handles variable spatial dims
    # Each program processes one (batch_idx, channel) pair
    
    # But with num_programs = num_channels, we can't process all batches
    # Let's change the grid to be batch_size * num_channels
    
    pass


# Simpler and more correct approach: compute the fused operation per-element
# Since adaptive avg pool to (1,1) averages ALL spatial elements,
# we can compute: out[b, c] = relu( (avg(b,c) - mean[c]) / sqrt(var[c] + eps) * weight[c] + bias[c] )
# where avg(b,c) = sum over h,w of input[b,c,h,w] / (H*W)

# Let's implement this properly using a different kernel design


def compute_adaptive_avg_pool2d_batch_norm_relu_fused(
    input: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-05,
    momentum: float = 0.1,
) -> torch.Tensor:
    """
    Fused kernel for:
    1. Adaptive average pooling to (1, 1)
    2. Batch normalization using running statistics
    3. ReLU activation
    
    Args:
        input: (batch, channels, H, W)
        running_mean: (channels,)
        running_var: (channels,)
        bias: (channels,)
        weight: (channels,)
        eps: epsilon for numerical stability
        momentum: momentum for BN (not used when training=False)
    
    Returns:
        output: (batch, channels, 1, 1)
    """
    batch_size, num_channels, H, W = input.shape
    output = torch.zeros(batch_size, num_channels, 1, 1, device=input.device, dtype=input.dtype)
    
    # Compute adaptive avg pool: average over spatial dimensions
    # For each (batch, channel), compute mean over H*W spatial positions
    # Then apply BN and ReLU
    
    # Reshape input to (batch, channels, H*W)
    input_flat = input.view(batch_size, num_channels, H * W)
    
    # Compute sum over spatial dimensions
    # Using triton for the reduction
    BLOCK_SIZE = min(8192, H * W)
    
    # For each (batch, channel), compute:
    # 1. Sum over spatial dims
    # 2. Divide by total to get average
    # 3. Apply BN: (avg - mean) / sqrt(var + eps) * weight + bias
    # 4. Apply ReLU: max(0, result)
    
    grid = (batch_size * num_channels,)
    
    fused_kernel[grid](
        input_flat,
        running_mean,
        running_var,
        bias,
        weight,
        output.squeeze(-1).squeeze(-1),  # (batch, channels)
        batch_size,
        num_channels,
        H * W,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output back to (batch, channels, 1, 1)
    output = output.view(batch_size, num_channels, 1, 1)
    
    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 2}),
    ],
    key=['num_spatial'],
)
@triton.jit
def fused_kernel(
    input_ptr,  # (batch, channels, H*W) flattened
    running_mean_ptr,  # (channels,)
    running_var_ptr,  # (channels,)
    bias_ptr,  # (channels,)
    weight_ptr,  # (channels,)
    output_ptr,  # (batch, channels) output for each b,c pair
    batch_size: tl.constexpr,
    num_channels: tl.constexpr,
    num_spatial: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes one (batch_idx, channel) pair.
    It sums all spatial positions, computes average, then applies BN and ReLU.
    """
    # Program ID maps to (batch_idx, channel)
    pid = tl.program_id(0)
    batch_idx = pid // num_channels
    channel_idx = pid % num_channels
    
    # Compute the offset for this batch, channel in the flattened input
    # input is (batch, channels, num_spatial)
    base_offset = batch_idx * num_channels * num_spatial + channel_idx * num_spatial
    
    # Sum over spatial dimension
    sum_val = 0.0
    for i in range(0, num_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_spatial
        
        # Load input values
        vals = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
    
    # Compute average
    avg = sum_val / num_spatial
    
    # Load BN parameters
    mean = tl.load(running_mean_ptr + channel_idx)
    var = tl.load(running_var_ptr + channel_idx)
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Apply BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = tl.rsqrt(var + eps)
    normalized = (avg - mean) * inv_std * weight + bias
    
    # Apply ReLU: max(0, normalized)
    output_val = tl.maximum(normalized, 0.0)
    
    # Store output
    output_offset = batch_idx * num_channels + channel_idx
    tl.store(output_ptr + output_offset, output_val)


def fused_adaptive_avg_pool2d_batch_norm_relu(
    input: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-05,
    momentum: float = 0.1,
) -> torch.Tensor:
    """
    Wrapper for the fused kernel.
    """
    batch_size, num_channels, H, W = input.shape
    
    # For small spatial sizes, the overhead of Triton might not be worth it
    # But for larger spatial sizes, the fusion saves memory bandwidth
    
    # Flatten spatial dimensions
    input_flat = input.reshape(batch_size, num_channels, H * W)
    
    # Allocate output
    output = torch.zeros(batch_size, num_channels, device=input.device, dtype=input.dtype)
    
    # Compute number of programs
    num_programs = batch_size * num_channels
    
    # Select block size based on spatial size
    if H * W >= 4096:
        BLOCK_SIZE = 4096
    elif H * W >= 2048:
        BLOCK_SIZE = 2048
    elif H * W >= 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 512
    
    # Launch kernel
    grid = (num_programs,)
    
    fused_kernel[grid](
        input_flat,
        running_mean,
        running_var,
        bias,
        weight,
        output,
        batch_size,
        num_channels,
        H * W,
        1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to (batch, channels, 1, 1)
    output = output.reshape(batch_size, num_channels, 1, 1)
    
    return output


@torch.fx.wrap
def fused_adaptive_avg_pool2d_batch_norm_relu_wrap(
    in_5: torch.Tensor,
    in_1: torch.Tensor,
    in_2: torch.Tensor,
    in_3: torch.Tensor,
    in_4: torch.Tensor,
) -> torch.Tensor:
    """
    FX-wrapped version of the fused kernel.
    
    Args:
        in_5: input tensor (batch, channels, H, W)
        in_1: running_mean (channels,)
        in_2: running_var (channels,)
        in_3: bias (channels,)
        in_4: weight (channels,)
    
    Returns:
        output: (batch, channels, 1, 1) after adaptive_avg_pool2d + batch_norm + relu
    """
    return fused_adaptive_avg_pool2d_batch_norm_relu(
        in_5, in_1, in_2, in_3, in_4,
        eps=1e-05,
        momentum=0.1,
    )


def replacement_func():
    """
    Returns the replacement function.
    """
    return fused_adaptive_avg_pool2d_batch_norm_relu_wrap