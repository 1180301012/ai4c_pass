import torch
import triton
import triton.language as tl

def pattern(tensor_10, tensor_14):
    # Match the chunk operations
    chunk_10 = tensor_10.chunk(2, dim=1)
    tmp_16 = chunk_10[0]
    tmp_17 = chunk_10[1]
    
    chunk_14 = tensor_14.chunk(2, dim=1)
    tmp_19 = chunk_14[0]
    tmp_20 = chunk_14[1]
    
    return tmp_16, tmp_17, tmp_19, tmp_20

def replacement_args(tensor_10, tensor_14):
    return (tensor_10, tensor_14)

@triton.jit
def chunk_slice_kernel(
    input_ptr,
    output1_ptr,
    output2_ptr,
    batch_size, input_channels, height, width,
    split_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total elements in the tensor
    total_elements = batch_size * input_channels * height * width
    mask = offsets < total_elements
    
    if mask:
        # Calculate indices for each dimension
        x = offsets % width
        y = (offsets // width) % height
        f = (offsets // (width * height)) % input_channels
        b = offsets // (width * height * input_channels)
        
        # Load input value
        input_pos = ((b * input_channels + f) * height + y) * width + x
        input_val = tl.load(input_ptr + input_pos, mask=mask, other=0.0)
        
        # For first chunk (first half of channels)
        if f < split_dim_size:
            output1_pos = ((b * split_dim_size + f) * height + y) * width + x
            tl.store(output1_ptr + output1_pos, input_val, mask=mask)
        
        # For second chunk (second half of channels)  
        if f >= split_dim_size:
            output2_f = f - split_dim_size
            output2_pos = ((b * split_dim_size + output2_f) * height + y) * width + x
            tl.store(output2_ptr + output2_pos, input_val, mask=mask)

@torch.fx.wrap
def optimized_chunk_operations(tensor_10, tensor_14):
    # Process first tensor: [N, 40, H, W] -> [N, 20, H, W], [N, 20, H, W]
    shape_10 = tensor_10.shape
    batch_size_10 = shape_10[0]
    input_channels_10 = shape_10[1]
    height_10 = shape_10[2]
    width_10 = shape_10[3]
    split_size_10 = input_channels_10 // 2
    
    out_shape_16 = (batch_size_10, split_size_10, height_10, width_10)
    out_16 = torch.empty(out_shape_16, dtype=tensor_10.dtype, device=tensor_10.device)
    out_17 = torch.empty(out_shape_16, dtype=tensor_10.dtype, device=tensor_10.device)
    
    if tensor_10.numel() > 0:
        BLOCK_SIZE = 1024
        total_elements = tensor_10.numel()
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        chunk_slice_kernel[(num_programs,)](
            input_ptr=tensor_10,
            output1_ptr=out_16,
            output2_ptr=out_17,
            batch_size=batch_size_10,
            input_channels=input_channels_10,
            height=height_10,
            width=width_10,
            split_dim_size=split_size_10,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Process second tensor: [N, 80, H, W] -> [N, 40, H, W], [N, 40, H, W]
    shape_14 = tensor_14.shape
    batch_size_14 = shape_14[0]
    input_channels_14 = shape_14[1]
    height_14 = shape_14[2]
    width_14 = shape_14[3]
    split_size_14 = input_channels_14 // 2
    
    out_shape_19 = (batch_size_14, split_size_14, height_14, width_14)
    out_19 = torch.empty(out_shape_19, dtype=tensor_14.dtype, device=tensor_14.device)
    out_20 = torch.empty(out_shape_19, dtype=tensor_14.dtype, device=tensor_14.device)
    
    if tensor_14.numel() > 0:
        total_elements = tensor_14.numel()
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        chunk_slice_kernel[(num_programs,)](
            input_ptr=tensor_14,
            output1_ptr=out_19,
            output2_ptr=out_20,
            batch_size=batch_size_14,
            input_channels=input_channels_14,
            height=height_14,
            width=width_14,
            split_dim_size=split_size_14,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out_16, out_17, out_19, out_20

def replacement_func():
    return optimized_chunk_operations