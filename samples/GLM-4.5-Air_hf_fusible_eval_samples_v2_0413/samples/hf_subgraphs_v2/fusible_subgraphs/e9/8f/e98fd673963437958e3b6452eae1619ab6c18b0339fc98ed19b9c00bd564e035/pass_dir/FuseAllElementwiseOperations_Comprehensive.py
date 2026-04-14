import torch
import triton
import triton.language as tl

@triton.jit
def comprehensive_elementwise_fusion_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    n_elements,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_0_val = tl.load(in_0_ptr + offsets, mask=mask)
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask)
    
    # For in_2 (shape [1, 1, 2048]), we need to compute sigmoid on channel dimension
    # and broadcast to spatial dimensions
    channel_idx = (offsets % (n_channels)) // (1)  # Extract channel index
    in_2_channel_offset = channel_idx  # in_2 has shape [1, 1, 2048]
    
    # Load sigmoid values (compute sigmoid with exponential approximation for performance)
    # Use offset % n_channels to get the right channel, in_2_ptr + in_2_channel_offset
    in_2_val = tl.load(in_2_ptr + in_2_channel_offset, mask=tl.arange(n_channels) < n_channels)
    
    # Fused computation: (in_1 * sigmoid(in_2)) + in_0, then apply ReLU
    sigmoid_val = 1.0 / (1.0 + tl.exp(-in_2_val))  # Sigmoid for the channel values
    
    # Broadcast sigmoid to spatial dimensions (implicit through memory access pattern)
    broadcast_sigmoid = sigmoid_val  # Will be broadcasted by compiler through register reuse
    
    # Fused multiply-add: result = (in_1 * sigmoid(in_2)) + in_0
    multiply_val = in_1_val * broadcast_sigmoid
    add_val = multiply_val + in_0_val
    
    # Apply ReLU: max(0, x)
    result = tl.maximum(add_val, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def comprehensive_elementwise_fusion(in_0, in_1, in_2):
    """Fuses all element-wise operations into single高效 kernel"""
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor 
    out = torch.empty_like(in_0)
    
    # Launch Triton kernel
    comprehensive_elementwise_fusion_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2.flatten(),  # Flatten in_2 to [2048] for easier indexing
        out_ptr=out,
        n_elements=N,
        n_channels=in_2.shape[-1],  # 2048 channels
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, in_0, in_1, in_2):
    # Match the entire element-wise computation from model.py:
    # tmp_0 = in_2.sigmoid()
    # tmp_1 = tmp_0.view(1, -1, 1, 1)  
    # tmp_2 = tmp_1.expand_as(in_1)
    # tmp_3 = in_1 * tmp_2
    # tmp_3 += in_0
    # tmp_4 = tmp_3
    # tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    
    sigmoid_val = in_2.sigmoid()
    view_op = sigmoid_val.view(1, -1, 1, 1)
    expand_op = view_op.expand_as(in_1)
    multiply_result = in_1 * expand_op
    # Use in-place add to match the actual graph exactly
    tmp_3 = multiply_result
    tmp_3 += in_0  # In-place add operation
    add_result = tmp_3
    relu_result = torch.nn.functional.relu(add_result, inplace=True)
    
    return relu_result

def replacement_args(tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@torch.fx.wrap
def comprehensive_elementwise_fusion(in_0, in_1, in_2):
    """Fuses all element-wise operations into single高效 kernel"""
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor 
    out = torch.empty_like(in_0)
    
    # Launch Triton kernel
    comprehensive_elementwise_fusion_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2.flatten(),  # Flatten in_2 to [2048] for easier indexing
        out_ptr=out,
        n_elements=N,
        n_channels=in_2.shape[-1],  # 2048 channels
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return comprehensive_elementwise_fusion