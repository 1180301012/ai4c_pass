import torch
import triton
import triton.language as tl


def pattern(conv_out, in_2):
    """
    Match the pattern: sigmoid(conv_out) -> view(1, -1, 1, 1) -> multiply with in_2
    
    This pattern represents the fusion of activation, reshape, and multiplication operations
    that appear sequentially in the target computation graphs.
    """
    tmp_3 = torch.sigmoid(conv_out)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    return tmp_5


def replacement_args(conv_out, in_2):
    """
    Extract arguments needed for the replacement kernel
    """
    return (conv_out, in_2)


@triton.jit
def fused_sigmoid_broadcast_mul_kernel(
    out_ptr,
    conv_out_ptr,
    in_2_ptr,
    batch_size,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs channel-wise gating with proper broadcasting.
    
    The kernel:
    1. Applies sigmoid to conv_out ([batch_size, out_channels, 1, 1])
    2. Broadcasts sigmoid result to match in_2 ([batch_size, out_channels, height, width])
    3. Performs element-wise multiplication
    
    Each thread handles one element from in_2 and loads the appropriate conv_out value by channel.
    """
    pid = tl.program_id(0)
    total_elements = batch_size * out_channels * height * width
    num_programs = tl.cdiv(total_elements, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
    
    # Calculate element offsets for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load in_2 values
    in_2_vals = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    
    # For each output element, determine the corresponding conv_out value by channel
    # conv_out has shape [batch_size, out_channels, 1, 1], so we need to extract the right channel
    
    # Calculate indices for the current element
    element_idx = offsets
    batch_idx = element_idx // (out_channels * height * width)
    remaining = element_idx % (out_channels * height * width)
    channel_idx = remaining // (height * width)
    
    # Calculate conv_out index (only batch and channel matter, spatial is [0,0])
    conv_out_idx = batch_idx * out_channels + channel_idx
    
    # Load the corresponding conv_out value for this channel
    conv_out_idx_valid = conv_out_idx < (batch_size * out_channels)
    conv_out_val = tl.load(conv_out_ptr + conv_out_idx, mask=conv_out_idx_valid, other=0.0)
    
    # Apply sigmoid to conv_out with proper dtype handling
    conv_out_fp32 = conv_out_val.to(tl.float32)
    
    # Sigmoid computation with numerical stability
    x_clamped = tl.where(conv_out_fp32 > 50.0, 50.0, tl.where(conv_out_fp32 < -50.0, -50.0, conv_out_fp32))
    sigmoid_out = 1.0 / (1.0 + tl.exp(-x_clamped))
    
    # Convert back to original dtype
    sigmoid_original_dtype = sigmoid_out.to(conv_out_val.type)
    
    # Apply the channel-wise gating
    result = sigmoid_original_dtype * in_2_vals
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_sigmoid_broadcast_mul(conv_out, in_2):
    """
    Fusion of sigmoid + element-wise multiplication using Triton kernel.
    
    The broadcast is handled by expanding conv_out to match in_2's spatial dimensions
    for efficient computation in the Triton kernel.
    """
    # Get tensor shapes
    batch_size, out_channels, height, width = in_2.shape
    
    # Create output tensor with same shape as in_2
    out = torch.empty_like(in_2)
    
    # Use fixed optimal block size for all tensor sizes
    total_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 1024  # Optimal block size for most workloads
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with simple 1D grid
    fused_sigmoid_broadcast_mul_kernel[(num_programs,)](
        out_ptr=out,
        conv_out_ptr=conv_out,
        in_2_ptr=in_2,
        batch_size=batch_size,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    """
    Return the fused function as a replacement
    """
    return fused_sigmoid_broadcast_mul