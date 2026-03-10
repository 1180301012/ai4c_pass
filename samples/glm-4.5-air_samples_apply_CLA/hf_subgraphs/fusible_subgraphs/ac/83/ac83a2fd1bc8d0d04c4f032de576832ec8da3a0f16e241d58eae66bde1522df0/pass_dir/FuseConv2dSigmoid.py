import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.sigmoid()
    return tmp_3

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def simple_conv1x1_sigmoid_kernel(
    x_ptr,       # Input: [batch, in_channels, 1, 1]
    weight_ptr,  # Weights: [out_channels, in_channels, 1, 1]
    bias_ptr,    # Bias: [out_channels]
    out_ptr,     # Output: [batch, out_channels, 1, 1]
    out_channels,
    in_channels,
    batch_size,
):
    # Each program handles one output channel
    pid = tl.program_id(0)
    
    if pid >= out_channels or batch_size == 0:
        return
    
    # Load bias
    bias = tl.load(bias_ptr + pid)
    
    # Vectorized computation for all batches
    # For this output channel, compute for each batch
    batch_idx = 0
    while batch_idx < batch_size:
        # Input offset for current batch
        input_offset = batch_idx * in_channels
        
        # Convolution computation
        conv_sum = bias
        for c in range(0, in_channels, 1):
            weight_val = tl.load(weight_ptr + pid * in_channels + c)
            input_val = tl.load(x_ptr + input_offset + c)
            conv_sum += weight_val * input_val
        
        # Apply sigmoid using Triton operations
        exp_val = tl.exp(-conv_sum)
        sigmoid_val = exp_val / (1.0 + exp_val)
        
        # Store result
        out_offset = batch_idx * out_channels + pid
        tl.store(out_ptr + out_offset, sigmoid_val)
        
        batch_idx += 1

@torch.fx.wrap  
def simple_conv2d_sigmoid(in_3, in_1, in_0):
    # Use the fact that this is essentially a linear transformation
    batch_size, in_channels, height, width = in_3.shape
    
    # Flatten to [batch_size, in_channels]
    x_flat = in_3.reshape(batch_size, in_channels)
    
    # Reshape weights to [out_channels, in_channels]
    w_flat = in_1.reshape(in_0.shape[0], in_channels)
    
    # Create output buffer on GPU
    result = torch.empty((batch_size, in_0.shape[0]), device=in_3.device, dtype=in_3.dtype)
    
    # Configure and launch kernel
    grid = (in_0.shape[0],)
    
    simple_conv1x1_sigmoid_kernel[grid](
        x_ptr=x_flat,
        weight_ptr=w_flat,
        bias_ptr=in_0,
        out_ptr=result,
        out_channels=in_0.shape[0],
        in_channels=in_channels,
        batch_size=batch_size,
    )
    
    # Reshape back to original format
    return result.reshape(batch_size, in_0.shape[0], height, width)

def replacement_func():
    return simple_conv2d_sigmoid