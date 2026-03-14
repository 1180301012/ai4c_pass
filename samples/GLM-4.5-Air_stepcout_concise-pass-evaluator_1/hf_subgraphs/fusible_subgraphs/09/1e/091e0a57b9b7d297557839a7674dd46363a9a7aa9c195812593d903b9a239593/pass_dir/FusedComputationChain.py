import torch
import triton
import triton.language as tl

@triton.jit
def chained_computation_kernel(
    sigmoid_input_ptr,
    x1_ptr,
    x0_ptr,
    output_ptr,
    N: tl.constexpr,  # batch size (1)
    C: tl.constexpr,  # channels (2048)
    H: tl.constexpr,  # spatial height
    W: tl.constexpr,  # spatial width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    channel_pid = tl.program_id(0)
    
    # Load sigmoid input for this channel: [1, 1, C] -> extract channel
    sigmoid_value = tl.load(sigmoid_input_ptr + channel_pid)
    
    # Compute sigmoid and reshape: [1, 1, C] -> [1, C, 1, 1]
    sigmoid_reshaped = tl.where(sigmoid_value >= 0, 1 / (1 + tl.exp(-sigmoid_value)), tl.exp(sigmoid_value) / (1 + tl.exp(sigmoid_value)))
    
    # Load spatial data for this channel: [C, H, W] -> extract (H, W) for this channel
    channel_offset = channel_pid * H * W
    
    total_spatial_elements = H * W
    offsets = channel_offset + tl.arange(0, total_spatial_elements)
    mask = offsets < (C * H * W)
    
    x1_data = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x0_data = tl.load(x0_ptr + offsets, mask=mask, other=0.0)
    
    # Expand sigmoid across spatial dimensions: [1, C, 1, 1] -> [1, C, H, W]
    # For spatial elements, use the sigmoid value for this channel
    sigmoid_expanded = tl.where(offsets < (channel_pid + 1) * total_spatial_elements and 
                                offsets >= channel_pid * total_spatial_elements,
                                sigmoid_reshaped, 1.0)
    
    # Fused operations: sigmoid * x1 + x0
    intermediate = sigmoid_expanded * x1_data + x0_data
    
    # Apply ReLU
    relu_output = tl.maximum(intermediate, 0.0)
    
    # Compute spatial mean and flatten
    channel_sum = tl.sum(relu_output)
    channel_mean = channel_sum / total_spatial_elements
    
    # Store final result: [1, C]
    tl.store(output_ptr + channel_pid, channel_mean)

@torch.fx.wrap
def fused_computation_chain(x2, x1, x0):
    N, C, H, W = x1.shape
    
    output = torch.empty((N, C), dtype=x1.dtype, device=x1.device)
    
    # Process each sample in batch
    for i in range(N):
        grid = (C,)  # Each block handles one channel
        chained_computation_kernel[grid](
            sigmoid_input_ptr=x2[i].squeeze(),  # [1, 1, C] -> [C]
            x1_ptr=x1[i],
            x0_ptr=x0[i],
            output_ptr=output[i],
            N=N,
            C=C,
            H=H,
            W=W,
            BLOCK_SIZE=1,
        )
    
    return output

def pattern(in_2, in_1, in_0):
    """Match the complete computation chain from inputs to final output"""
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_3 = tmp_3 + in_0
    tmp_4 = tmp_3
    tmp_5 = torch.nn.functional.relu(tmp_4)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

def replacement_func():
    return fused_computation_chain