import torch
import triton
import triton.language as tl


# Simple autotune with good defaults
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
    ],
    key=['N', 'OC', 'flattened_size'],
)
@triton.jit
def fused_conv2d_flatten_kernel(
    # Conv2d inputs
    input_ptr,
    weight_ptr,
    bias_ptr,
    # Output pointer (flattened result)
    output_ptr,
    # Output dimensions
    N,
    OC,
    flattened_size,
    # Input dimensions
    IC,
    IH,
    IW,
    # Spatial dimensions
    OH,
    OW,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv2d (1x1 kernel) + Flatten kernel.
    Uses 2D grid where each program handles one (batch, out_channel) pair.
    """
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    
    # Bounds check
    if pid_n >= N or pid_oc >= OC:
        return
    
    # Offsets for vectorized load/store
    offs = tl.arange(0, BLOCK_SIZE)
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + pid_oc)
    
    # Initialize accumulator for this chunk
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over spatial positions in chunks of BLOCK_SIZE
    for spatial_start in range(0, flattened_size, BLOCK_SIZE):
        # Compute mask for valid positions
        spatial_offs = spatial_start + offs
        mask = spatial_offs < flattened_size
        
        # Compute h, w for each position
        h = spatial_offs // OW
        w = spatial_offs % OW
        
        # Process all IC channels
        for ic_idx in range(IC):
            # Load weight for this input channel
            weight_offset = pid_oc * IC + ic_idx
            weight_val = tl.load(weight_ptr + weight_offset)
            
            # Compute input offsets: input[n, ic, h, w]
            input_offsets = (pid_n * IC + ic_idx) * (IH * IW) + h * IW + w
            input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
            
            # Multiply and accumulate
            result += input_vals * weight_val
        
        # Store results (add bias once per chunk)
        output_base = pid_n * (OC * flattened_size) + pid_oc * flattened_size + spatial_start
        output_offsets = output_base + offs
        tl.store(output_ptr + output_offsets, result + bias_val, mask=mask)
        
        # Reset accumulator for next chunk
        result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)


@torch.fx.wrap
def fused_conv2d_flatten_wrapper(bias_tensor, weight_tensor, input_tensor):
    """
    Wrapper for fused Conv2d + Flatten operation.
    Hardcoded for: stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    Input: in_0=bias, in_1=weight, in_2=input
    """
    N, IC, IH, IW = input_tensor.shape
    OC, IC_w, KH, KW = weight_tensor.shape
    
    # Fixed parameters from the pattern
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    
    # Compute output spatial dimensions (for 1x1 conv with stride=1, pad=0)
    OH = (IH + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    OW = (IW + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1
    flattened_size = OH * OW
    
    # Create output tensor: [N, OC, OH*OW]
    output = torch.empty((N, OC, flattened_size), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Simple grid: (N, OC)
    grid = (N, OC)
    
    fused_conv2d_flatten_kernel[grid](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        N, OC, flattened_size,
        IC, IH, IW,
        OH, OW,
        128,  # BLOCK_SIZE
    )
    
    return output


def pattern(in_0, in_1, in_2):
    """
    Match the Conv2d + Flatten pattern.
    Conv2d args: (input, weight, bias, stride, padding, dilation, groups)
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_conv2d_flatten_wrapper