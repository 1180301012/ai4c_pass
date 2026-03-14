import torch
import triton
import triton.language as tl

@triton.jit
def fused_relu_adaptive_avg_pool_kernel(
    input_ptr,
    output_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    pid = tl.program_id(0)  # channel id
    total_elements = H * W
    
    # Load one channel data: [H, W]
    channel_offset = pid * total_elements
    offsets = channel_offset + tl.arange(0, total_elements)
    mask = offsets < (C * total_elements)
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    relu_output = tl.maximum(input_data, 0.0)
    
    # Compute mean by summing and dividing by H*W
    # Since we have warp-level reductions, we'll implement a simple sum
    # followed by division
    channel_sum = tl.sum(relu_output)
    channel_mean = channel_sum / total_elements
    
    # Store result at output position pid
    tl.store(output_ptr + pid, channel_mean)

@torch.fx.wrap
def fused_relu_adaptive_avg_pool(input_tensor):
    N, C, H, W = input_tensor.shape
    output = torch.empty((N, C), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Handle batch dimension N by processing each sample separately
    for i in range(N):
        BLOCK_SIZE = C  # Each block handles one channel
        grid = (C,)
        
        fused_relu_adaptive_avg_pool_kernel[grid](
            input_ptr=input_tensor[i],
            output_ptr=output[i],
            C=C,
            H=H,
            W=W,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

def pattern(x):
    """Match the pattern: relu -> adaptive_avg_pool2d -> flatten"""
    tmp_5 = torch.nn.functional.relu(x, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(x):
    return (x,)

def replacement_func():
    return fused_relu_adaptive_avg_pool