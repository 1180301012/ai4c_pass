import torch
import triton
import triton.language as tl

def pattern(tensor_list, dim):
    # Match: torch.cat + .to(dtype=torch.float16)
    tmp_6 = torch.cat(tensor_list, dim=dim)
    tmp_7 = tmp_6.to(dtype=torch.float16)
    return tmp_7

def replacement_args(tensor_list, dim):
    return (tensor_list, dim)

@triton.jit
def fused_cat_convert_kernel(
    ptr_list,
    output_ptr,
    strides_list,
    tensor_shapes,
    total_elements,
    output_stride_batch,
    output_stride_channels,
    output_stride_height, 
    output_stride_width,
    block_size: tl.constexpr,
):
    program_id = tl.program_id(0)
    
    block_start = program_id * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Initialize output
    output_vals = tl.zeros([block_size], dtype=tl.float16)
    
    # Process each tensor in the list
    for tensor_idx in range(len(ptr_list)):
        tensor_ptr = ptr_list[tensor_idx]
        tensor_shape = tensor_shapes[tensor_idx]
        tensor_stride = strides_list[tensor_idx]
        
        # Calculate if this offset belongs to this tensor
        tensor_elements = tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * tensor_shape[3]
        tensor_start_idx = 0
        if tensor_idx > 0:
            prev_shapes = tensor_shapes[:tensor_idx]
            tensor_start_idx = sum(prev_shape[0] * prev_shape[1] * prev_shape[2] * prev_shape[3] for prev_shape in prev_shapes)
        
        tensor_mask = (offsets >= tensor_start_idx) & (offsets < tensor_start_idx + tensor_elements)
        final_mask = mask & tensor_mask
        
        if tl.any(final_mask):
            # Calculate relative offset within this tensor
            rel_offset = offsets - tensor_start_idx
            
            # Convert flat offset to tensor coordinates
            batch_idx = rel_offset // (tensor_shape[1] * tensor_shape[2] * tensor_shape[3])
            remainder = rel_offset % (tensor_shape[1] * tensor_shape[2] * tensor_shape[3])
            channels_idx = remainder // (tensor_shape[2] * tensor_shape[3])
            remainder = remainder % (tensor_shape[2] * tensor_shape[3])
            height_idx = remainder // tensor_shape[3]
            width_idx = remainder % tensor_shape[3]
            
            # Calculate input position
            input_pos = (tensor_ptr + 
                        batch_idx * tensor_stride[0] + 
                        channels_idx * tensor_stride[1] + 
                        height_idx * tensor_stride[2] + 
                        width_idx * tensor_stride[3])
            
            # Load and convert
            input_vals = tl.load(input_pos, mask=final_mask, other=0.0)
            output_vals = tl.where(final_mask, input_vals.to(tl.float16), output_vals)
    
    # Store output
    output_pos = output_ptr + offsets
    tl.store(output_pos, output_vals, mask=mask)

@torch.fx.wrap
def fused_cat_convert(tensor_list, dim):
    if dim != 0:
        # Fall back to original implementation for non-zero dimensions
        concatenated = torch.cat(tensor_list, dim=dim)
        return concatenated.to(dtype=torch.float16)
    
    # Calculate total elements in final tensor
    total_elements = 0
    tensor_shapes = []
    for tensor in tensor_list:
        shape = tensor.shape
        total_elements += shape[0] * shape[1] * shape[2] * shape[3]
        tensor_shapes.append(shape)
    
    # Create output tensor
    total_patches = sum(shape[0] for shape in tensor_shapes)
    output_shape = [total_patches, tensor_shapes[0][1], tensor_shapes[0][2], tensor_shapes[0][3]]
    output = torch.empty(output_shape, dtype=torch.float16, device=tensor_list[0].device)
    
    # Set up kernel configuration
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Prepare pointers and strides for each tensor
    ptr_list = [t.data_ptr() for t in tensor_list]
    strides_list = [list(t.stride()) for t in tensor_list]
    
    # Launch kernel
    fused_cat_convert_kernel[grid_size](
        ptr_list=ptr_list,
        output_ptr=output.data_ptr(),
        strides_list=strides_list,
        tensor_shapes=tensor_shapes,
        total_elements=total_elements,
        output_stride_batch=output.stride(0),
        output_stride_channels=output.stride(1),
        output_stride_height=output.stride(2),
        output_stride_width=output.stride(3),
        block_size=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_cat_convert