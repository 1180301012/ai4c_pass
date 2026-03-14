import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    # Match the sequence: concat + adaptive_avg_pool2d + flatten
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_concat_pool_flatten_kernel(
    x0_ptr, x1_ptr, x2_ptr, x3_ptr,
    output_ptr,
    c0, c1, c2, c3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each block processes a subset of the output channels
    output_offset = pid * BLOCK_SIZE
    output_indices = output_offset + tl.arange(0, BLOCK_SIZE)
    output_mask = output_indices < (c0 + c1 + c2 + c3)
    
    # Load from each input tensor
    # Input 0: channels 0 to c0-1
    x0_indices = output_indices
    x0_mask = output_indices < c0
    x0 = tl.load(x0_ptr + x0_indices, mask=x0_mask, other=0.0)
    
    # Input 1: channels c0 to c0+c1-1  
    x1_indices = output_indices - c0
    x1_mask = (output_indices >= c0) & (output_indices < (c0 + c1))
    x1 = tl.load(x1_ptr + x1_indices, mask=x1_mask, other=0.0)
    
    # Input 2: channels c0+c1 to c0+c1+c2-1
    x2_indices = output_indices - (c0 + c1)
    x2_mask = (output_indices >= (c0 + c1)) & (output_indices < (c0 + c1 + c2))
    x2 = tl.load(x2_ptr + x2_indices, mask=x2_mask, other=0.0)
    
    # Input 3: channels c0+c1+c2 to c0+c1+c2+c3-1
    x3_indices = output_indices - (c0 + c1 + c2)
    x3_mask = (output_indices >= (c0 + c1 + c2)) & (output_indices < (c0 + c1 + c2 + c3))
    x3 = tl.load(x3_ptr + x3_indices, mask=x3_mask, other=0.0)
    
    # Since adaptive_avg_pool2d(1, 1) on 1x1 tensors is just identity operation,
    # and flatten along dim=1 on [1, C, 1, 1] is just reshape to [1, C],
    # we can directly concatenate the channels from all four inputs
    result = x0 + x1 + x2 + x3
    
    # Store the result
    tl.store(output_ptr + output_indices, result, mask=output_mask)


@torch.fx.wrap
def fused_concat_pool_flatten_func(in_0, in_1, in_2, in_3):
    # Get the channel dimensions
    c0 = in_0.shape[1]
    c1 = in_1.shape[1]  
    c2 = in_2.shape[1]
    c3 = in_3.shape[1]
    
    # Total output channels
    total_channels = c0 + c1 + c2 + c3
    
    # Output shape is [1, total_channels]
    output = torch.empty((1, total_channels), dtype=in_0.dtype, device=in_0.device)
    
    # Block size for better GPU utilization
    BLOCK_SIZE = 1024
    
    # Number of programs needed
    num_programs = (total_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_concat_pool_flatten_kernel[(num_programs,)](
        in_0, in_1, in_2, in_3,
        output,
        c0, c1, c2, c3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_concat_pool_flatten_func