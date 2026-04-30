import torch
import triton
import triton.language as tl


@triton.jit
def bisenet_conv_sigmoid_kernel(
    # Input pointers
    in_5_ptr, weight_ptr, bias_ptr,
    # Output pointer
    out_ptr,
    # Shapes
    batch: tl.constexpr, channels: tl.constexpr,
    h: tl.constexpr, w: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs - 2D grid: (batch * channels, spatial)
    pid = tl.program_id(0)
    
    # Calculate which batch and channel this program handles
    spatial_size = h * w
    b = pid // channels
    c = pid % channels
    
    # Calculate spatial position for this program
    hw_start = tl.program_id(0) * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    
    # Masks for bounds checking
    hw_mask = hw_offsets < spatial_size
    
    # Calculate input base offset
    base_offset = b * channels * h * w + c * h * w
    
    # Load input values for this block
    h_indices = hw_offsets // w
    w_indices = hw_offsets % w
    in_offsets = base_offset + h_indices * w + w_indices
    
    # Perform 1x1 conv: accumulate over all input channels
    # For output channel c: out[c] = sum over k of inp[k] * weight[c, k] + bias[c]
    # Initialize with zeros of the correct type
    conv_out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for k in range(channels):
        # Load weight[c, k]
        weight_offset = c * channels + k
        w_val = tl.load(weight_ptr + weight_offset, mask=k < channels, other=0.0)
        # Load input[k] at same spatial position
        inp_k_offset = b * channels * h * w + k * h * w + h_indices * w + w_indices
        inp_k = tl.load(in_5_ptr + inp_k_offset, mask=hw_mask, other=0.0)
        # Expand w_val to block size and accumulate
        conv_out += inp_k * w_val
    
    # Add bias for output channel c
    bias_val = tl.load(bias_ptr + c, mask=c < channels, other=0.0)
    conv_out = conv_out + bias_val
    
    # Apply sigmoid
    result = 1.0 / (1.0 + tl.exp(-conv_out))
    
    # Store output
    out_offset = base_offset + h_indices * w + w_indices
    tl.store(out_ptr + out_offset, result, mask=hw_mask)


def pattern(in_0, in_1, in_2):
    """
    Match conv2d + sigmoid pattern from BiSeNetV2 BGA.
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for the conv+sigmoid kernel.
    """
    return (in_0, in_1, in_2)


@torch.fx.wrap
def bisenet_conv_sigmoid(in_0, in_1, in_2):
    """
    Fused conv2d + sigmoid kernel for BiSeNetV2 BGA.
    """
    batch, channels, h, w = in_2.shape
    
    # Output
    out = torch.empty_like(in_2)
    
    # Grid configuration - 1D: total number of output elements
    # Each program handles one channel's output values
    num_programs = batch * channels
    BLOCK_SIZE = 256  # Process 256 spatial elements per program
    
    bisenet_conv_sigmoid_kernel[(num_programs,)](
        in_2, in_1, in_0,
        out,
        batch, channels, h, w,
        BLOCK_SIZE
    )
    
    return out


def replacement_func():
    return bisenet_conv_sigmoid