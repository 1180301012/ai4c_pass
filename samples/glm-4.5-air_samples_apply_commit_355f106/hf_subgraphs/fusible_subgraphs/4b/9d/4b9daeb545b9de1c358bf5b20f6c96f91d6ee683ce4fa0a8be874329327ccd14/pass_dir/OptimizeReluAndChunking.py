import torch
import triton
import triton.language as tl

def pattern(relu_input, tensor_40_channels, tensor_80_channels):
    tmp_2 = torch.nn.functional.relu(relu_input, inplace=False)
    tmp_3 = tensor_40_channels.chunk(2, dim=1)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_2.chunk(2, dim=1)
    tmp_7 = tmp_6[0]
    tmp_8 = tmp_6[1]
    return (tmp_4, tmp_7, tmp_5, tmp_8)

def replacement_args(relu_input, tensor_40_channels, tensor_80_channels):
    return (relu_input, tensor_40_channels, tensor_80_channels)

@triton.jit
def optimized_relu_chunk_kernel(
    relu_input_ptr,
    tensor40_ptr,
    tensor80_split_ptr_1,
    tensor80_split_ptr_2,
    tensor40_split_ptr_1,
    tensor40_split_ptr_2,
    n_elements80_1,
    n_elements80_2,
    n_elements40_1,
    n_elements40_2,
    BLOCK_SIZE: tl.constexpr,
):
    # Process first half of the 80-channel tensor (relu input split)
    block_start_80_1 = tl.program_id(0) * BLOCK_SIZE
    offsets_80_1 = block_start_80_1 + tl.arange(0, BLOCK_SIZE)
    mask_80_1 = offsets_80_1 < n_elements80_1
    
    relu_input = tl.load(relu_input_ptr + offsets_80_1, mask=mask_80_1, other=0.0)
    relu_out = tl.maximum(relu_input, 0.0)
    tl.store(tensor80_split_ptr_1 + offsets_80_1, relu_out, mask=mask_80_1)
    
    # Process second half of the 80-channel tensor 
    block_start_80_2 = tl.program_id(1) * BLOCK_SIZE
    offsets_80_2 = block_start_80_2 + tl.arange(0, BLOCK_SIZE)
    mask_80_2 = offsets_80_2 < n_elements80_2
    
    relu_input_2 = tl.load(relu_input_ptr + offsets_80_2, mask=mask_80_2, other=0.0)
    relu_out_2 = tl.maximum(relu_input_2, 0.0)
    tl.store(tensor80_split_ptr_2 + offsets_80_2, relu_out_2, mask=mask_80_2)
    
    # Process first half of the 40-channel tensor (direct split)
    block_start_40_1 = tl.program_id(2) * BLOCK_SIZE
    offsets_40_1 = block_start_40_1 + tl.arange(0, BLOCK_SIZE)
    mask_40_1 = offsets_40_1 < n_elements40_1
    
    tensor40_1 = tl.load(tensor40_ptr + offsets_40_1, mask=mask_40_1, other=0.0)
    tl.store(tensor40_split_ptr_1 + offsets_40_1, tensor40_1, mask=mask_40_1)
    
    # Process second half of the 40-channel tensor
    block_start_40_2 = tl.program_id(3) * BLOCK_SIZE
    offsets_40_2 = block_start_40_2 + tl.arange(0, BLOCK_SIZE)
    mask_40_2 = offsets_40_2 < n_elements40_2
    
    tensor40_2 = tl.load(tensor40_ptr + offsets_40_2, mask=mask_40_2, other=0.0)
    tl.store(tensor40_split_ptr_2 + offsets_40_2, tensor40_2, mask=mask_40_2)

@torch.fx.wrap
def optimized_relu_chunk_forward(relu_input, tensor_40_channels):
    # Calculate sizes for each chunk
    tensor80_shape = relu_input.shape
    tensor40_shape = tensor_40_channels.shape
    
    # First half channels for 80-channel tensor (receptive field split)
    tensor80_1_size = (tensor80_shape[0] // 2) * tensor80_shape[1] * tensor80_shape[2] * tensor80_shape[3]
    # Second half channels for 80-channel tensor
    tensor80_2_size = tensor80_shape[0] * (tensor80_shape[1] // 2) * tensor80_shape[2] * tensor80_shape[3]
    
    # For 40-channel tensor (channel split)
    tensor40_1_size = tensor40_shape[0] // 2 * tensor40_shape[1] * tensor40_shape[2] * tensor40_shape[3]
    tensor40_2_size = tensor40_1_size  # Same size since we split into 2 equal parts
    
    BLOCK_SIZE = 1024
    
    # Calculate grid size for 4 parallel operations
    programs_80_1 = (tensor80_1_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    programs_80_2 = (tensor80_2_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    programs_40_1 = (tensor40_1_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    programs_40_2 = (tensor40_2_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    tensor80_1 = torch.empty(tensor80_shape[0] // 2, tensor80_shape[1], tensor80_shape[2], tensor80_shape[3], 
                            dtype=relu_input.dtype, device=relu_input.device)
    tensor80_2 = torch.empty(tensor80_shape[0], tensor80_shape[1] // 2, tensor80_shape[2], tensor80_shape[3], 
                            dtype=relu_input.dtype, device=relu_input.device)
    tensor40_1 = torch.empty(tensor40_shape[0] // 2, tensor40_shape[1], tensor40_shape[2], tensor40_shape[3], 
                            dtype=tensor_40_channels.dtype, device=tensor_40_channels.device)
    tensor40_2 = torch.empty(tensor40_shape[0] // 2, tensor40_shape[1], tensor40_shape[2], tensor40_shape[3], 
                            dtype=tensor_40_channels.dtype, device=tensor_40_channels.device)
    
    # Launch kernel with 4 program dimensions
    optimized_relu_chunk_kernel[(programs_80_1, programs_80_2, programs_40_1, programs_40_2)](
        relu_input_ptr=relu_input,
        tensor40_ptr=tensor_40_channels,
        tensor80_split_ptr_1=tensor80_1,
        tensor80_split_ptr_2=tensor80_2,
        tensor40_split_ptr_1=tensor40_1,
        tensor40_split_ptr_2=tensor40_2,
        n_elements80_1=tensor80_1_size,
        n_elements80_2=tensor80_2_size,
        n_elements40_1=tensor40_1_size,
        n_elements40_2=tensor40_2_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tensor40_1, tensor80_1, tensor40_2, tensor80_2

def replacement_func():
    return optimized_relu_chunk_forward