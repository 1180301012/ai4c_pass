import torch
import triton
import triton.language as tl

# Pattern matching function - matches the normalization operation
def pattern(in_1):
    tmp_0 = in_1.sum(dim = 2, keepdim = True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Optimized kernel for normalization along dimension 2
@triton.jit
def compute_sum_kernel(
    input_ptr,
    sum_ptr,
    n_batch,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one (batch, channel, height) combination
    pid = tl.program_id(0)
    
    # Calculate indices for batch and channel
    batch_idx = pid // (n_channels * height)
    channel_idx = (pid // height) % n_channels
    height_idx = pid % height
    
    # Compute offset in the input tensor for this position
    base_offset = batch_idx * n_channels * height * width + channel_idx * height * width + height_idx * width
    
    # Load all elements along dimension 2 (width dimension) for this position
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (base_offset + width)
    
    # Load the width slice and compute sum
    input_slice = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(input_slice)
    
    # Store the sum at the corresponding position in the sum tensor
    sum_offset = batch_idx * n_channels * height + channel_idx * height + height_idx
    tl.store(sum_ptr + sum_offset, sum_val)

@triton.jit
def normalize_kernel(
    input_ptr,
    sum_ptr,
    output_ptr,
    n_batch,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one element
    pid = tl.program_id(0)
    
    # Calculate indices for this element
    batch_idx = pid // (n_channels * height * width)
    remaining = pid % (n_channels * height * width)
    
    channel_idx = remaining // (height * width)
    remaining = remaining % (height * width)
    
    height_idx = remaining // width
    width_idx = remaining % width
    
    # Compute input offset
    input_offset = batch_idx * n_channels * height * width + channel_idx * height * width + height_idx * width + width_idx
    
    # Compute sum offset for this (batch, channel, height)
    sum_offset = batch_idx * n_channels * height + channel_idx * height + height_idx
    
    # Load input value and corresponding sum
    input_val = tl.load(input_ptr + input_offset)
    sum_val = tl.load(sum_ptr + sum_offset)
    
    # Compute normalized value (avoid division by zero)
    norm_sum = tl.math.max(sum_val, 1e-7)
    output_val = input_val / norm_sum
    
    # Store normalized value
    tl.store(output_ptr + input_offset, output_val)

@torch.fx.wrap
def normalize_along_dim2(in_1):
    """
    Normalize tensor along dimension 2 (last spatial dimension)
    This is equivalent to: in_1 / in_1.sum(dim=2, keepdim=True)
    """
    batch, channels, height, width = in_1.shape
    
    # Create sum tensor with shape [batch, channels, height]
    sum_shape = (batch, channels, height)
    sum_tensor = torch.empty(sum_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Create output tensor
    output = torch.empty_like(in_1)
    
    # Block size for parallel processing
    BLOCK_SIZE = min(1024, width)  # Don't exceed dimension size
    
    # Compute sums along dimension 2
    total_sum_elements = batch * channels * height
    num_sum_programs = (total_sum_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    compute_sum_kernel[(num_sum_programs,)](
        input_ptr=in_1,
        sum_ptr=sum_tensor,
        n_batch=batch,
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Broadcast sum tensor to match input shape [batch, channels, height, width]
    broadcast_sum = sum_tensor.unsqueeze(-1).expand_as(in_1)
    
    # Perform element-wise division
    output = in_1 / broadcast_sum.clamp(min=1e-7)
    
    return output

# Replacement function (MUST return function reference, not call)
def replacement_func():
    return normalize_along_dim2