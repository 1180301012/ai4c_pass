import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_hardswish_flatten_kernel(
    # Conv2D inputs
    input_ptr, weight_ptr, bias_ptr,
    # Output pointer  
    output_ptr,
    # Shape info
    batch_size, in_channels, out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv2D (1x1 with bias) + Hardswish + Flatten(1, -1) kernel.
    
    Output[b,c] = sum_k(input[b,k,0,0] * weight[c,k,0,0]) + bias[c]
    Then: result = x * clamp(x + 3, 0, 6) / 6
    """
    pid = tl.program_id(0)
    
    # Each program computes one output element
    batch_idx = pid // out_channels
    channel_idx = pid % out_channels
    
    if batch_idx >= batch_size:
        return
    
    # Compute base offsets
    # Input[b,k,0,0] -> flat offset = b * in_channels + k (for H=W=1)
    # Weight[c,k,0,0] -> flat offset = c * in_channels + k
    input_base = batch_idx * in_channels
    weight_base = channel_idx * in_channels
    
    # Load bias
    bias = tl.load(bias_ptr + channel_idx).to(tl.float32)
    acc = bias
    
    # Dot product: sum over k
    for k in range(0, in_channels, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        mask_k = k_offsets < in_channels
        
        # Load and accumulate
        input_vals = tl.load(input_ptr + input_base + k_offsets, mask=mask_k, other=0.0).to(tl.float32)
        weight_vals = tl.load(weight_ptr + weight_base + k_offsets, mask=mask_k, other=0.0).to(tl.float32)
        
        acc += tl.sum(input_vals * weight_vals)
    
    # Hardswish: clamp(x + 3, 0, 6) * x / 6
    relu6 = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    result = acc * relu6 / 6.0
    
    # Store
    output_offset = batch_idx * out_channels + channel_idx
    tl.store(output_ptr + output_offset, result)


@torch.fx.wrap
def fused_conv_hardswish_flatten(input_tensor, weight_tensor, bias_tensor):
    """
    Fused Conv2D (1x1 with bias) + Hardswish + Flatten(1, -1) kernel.
    
    Expected input shapes:
    - input_tensor: [batch, in_channels, 1, 1] or already flattened
    - weight_tensor: [out_channels, in_channels, 1, 1]
    - bias_tensor: [out_channels]
    """
    # Get dimensions from weight tensor (always 4D with H=W=1)
    out_channels = weight_tensor.shape[0]
    in_channels = weight_tensor.shape[1]
    
    # Determine batch size based on input tensor dimensionality
    input_ndim = input_tensor.dim()
    
    if input_ndim == 4:
        batch_size = input_tensor.shape[0]
    elif input_ndim == 3:
        batch_size = input_tensor.shape[0]
    elif input_ndim == 2:
        batch_size = input_tensor.shape[0]
    elif input_ndim == 1:
        total = input_tensor.shape[0]
        batch_size = total // in_channels
    else:
        batch_size = 1
    
    # Allocate output with final shape [batch, out_channels]
    output = torch.empty((batch_size, out_channels), 
                         dtype=input_tensor.dtype, 
                         device=input_tensor.device)
    
    # Optimized block size for this problem size
    # in_channels=960 works well with BLOCK_SIZE=256
    BLOCK_SIZE = 256
    num_programs = batch_size * out_channels
    
    conv2d_hardswish_flatten_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2):
    """
    Match: Conv2D + Hardswish + Flatten pattern
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv_hardswish_flatten