import torch
import triton
import triton.language as tl

def pattern(in_2, in_3, tmp_1, tmp_2):
    """Pattern to match concatenation followed by stacking"""  
    tmp_0 = torch.cat((in_2, in_3), 1)
    tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0])
    return tmp_3

def replacement_args(in_2, in_3, tmp_1, tmp_2):
    return (in_2, in_3, tmp_1, tmp_2)

@triton.jit
def simple_concat_kernel(
    in2_ptr, in3_ptr, out_ptr,
    batch_size, channels2, channels3, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple concatenation kernel"""
    pid = tl.program_id(0)
    total_elements = batch_size * (channels2 + channels3) * height * width
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    for idx in range(start_idx, end_idx):
        batch_idx = idx // ((channels2 + channels3) * height * width)
        remainder = idx % ((channels2 + channels3) * height * width)
        channel_idx = remainder // (height * width)
        h_idx = remainder % height // width
        w_idx = remainder % width
        
        if channel_idx < channels2:
            # Copy from in_2
            src_offset = batch_idx * channels2 * height * width + channel_idx * height * width + h_idx * width + w_idx
            val = tl.load(in2_ptr + src_offset)
        else:
            # Copy from in_3
            src_offset = batch_idx * channels3 * height * width + (channel_idx - channels2) * height * width + h_idx * width + w_idx
            val = tl.load(in3_ptr + src_offset)
        
        out_offset = batch_idx * (channels2 + channels3) * height * width + channel_idx * height * width + h_idx * width + w_idx
        tl.store(out_ptr + out_offset, val)

@triton.jit 
def stack_kernel(
    tensor0_ptr, tensor1_ptr, tensor2_ptr, out_ptr,
    batch_size, channels0, channels1, channels2,
    height, width,
    BLOCK_SIZE: tl.constexpr,
):
    """Stack three tensors along first dimension"""
    pid = tl.program_id(0)
    
    # Process elements in flattened format
    total_elements = batch_size * max(channels0, channels1, channels2) * height * width
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    for idx in range(start_idx, end_idx):
        # Calculate which tensor position this belongs to
        elements_per_tensor = batch_size * max(channels0, channels1, channels2) * height * width
        tensor_pos = idx // elements_per_tensor
        
        # Position within the tensor
        tensor_idx = idx % elements_per_tensor
        
        batch_idx = tensor_idx // (max(channels0, channels1, channels2) * height * width)
        remainder = tensor_idx % (max(channels0, channels1, channels2) * height * width)
        channel_idx = remainder // (height * width)
        h_idx = remainder % height // width  
        w_idx = remainder % width
        
        # Load from appropriate tensor
        if tensor_pos == 0:  # tmp_1
            if channel_idx < channels0:
                offset = batch_idx * channels0 * height * width + channel_idx * height * width + h_idx * width + w_idx
                val = tl.load(tensor0_ptr + offset)
            else:
                val = 0.0  # Pad with zeros if needed
        elif tensor_pos == 1:  # tmp_2
            if channel_idx < channels1:
                offset = batch_idx * channels1 * height * width + channel_idx * height * width + h_idx * width + w_idx  
                val = tl.load(tensor1_ptr + offset)
            else:
                val = 0.0  # Pad with zeros if needed
        else:  # concatenated tensor
            if channel_idx < channels2:
                offset = batch_idx * channels2 * height * width + channel_idx * height * width + h_idx * width + w_idx
                val = tl.load(tensor2_ptr + offset)
            else:
                val = 0.0  # Pad with zeros if needed
        
        # Store in output layout: [3, batch, max_channels, height, width]
        out_offset = tensor_pos * elements_per_tensor + tensor_idx
        tl.store(out_ptr + out_offset, val)

@torch.fx.wrap
def optimized_concat_stack(in_2, in_3, tmp_1, tmp_2):
    """Optimized concatenation + stacking function with pure Triton implementation"""
    batch_size = in_2.shape[0]
    channels2 = in_2.shape[1]  # Should match in_3.shape[1]
    channels3 = in_3.shape[1]  # Should match in_2.shape[1] 
    channels1 = tmp_1.shape[1]  # Should match tmp_2.shape[1]
    height = in_2.shape[2]  # Should all be 40
    width = in_2.shape[3]   # Should all be 40
    
    # Verify tensor shapes
    assert tmp_1.shape[0] == batch_size
    assert tmp_1.shape[2] == height
    assert tmp_1.shape[3] == width
    assert tmp_2.shape == tmp_1.shape
    
    # Step 1: Concatenate in_2 and in_3
    concat_channels = channels2 + channels3
    concat_out = torch.empty((batch_size, concat_channels, height, width), 
                           dtype=in_2.dtype, device=in_2.device)
    
    # Launch concatenation kernel
    BLOCK_SIZE = 1024
    concat_total_elements = batch_size * concat_channels * height * width
    concat_grid_size = (concat_total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    simple_concat_kernel[concat_grid_size](
        in_2, in_3, concat_out,
        batch_size, channels2, channels3, height, width,
        BLOCK_SIZE
    )
    
    # Step 2: Stack the three tensors
    max_channels = max(channels1, concat_channels)
    out_shape = (3, batch_size, max_channels, height, width)
    result = torch.empty(out_shape, dtype=tmp_1.dtype, device=tmp_1.device)
    
    # Launch stacking kernel
    stack_total_elements = batch_size * max_channels * height * width * 3  # 3 tensors
    stack_grid_size = (stack_total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    stack_kernel[stack_grid_size](
        tmp_1, tmp_2, concat_out, result,
        batch_size, channels1, tmp_2.shape[1], concat_channels,
        height, width,
        BLOCK_SIZE
    )
    
    return result

def replacement_func():
    return optimized_concat_stack