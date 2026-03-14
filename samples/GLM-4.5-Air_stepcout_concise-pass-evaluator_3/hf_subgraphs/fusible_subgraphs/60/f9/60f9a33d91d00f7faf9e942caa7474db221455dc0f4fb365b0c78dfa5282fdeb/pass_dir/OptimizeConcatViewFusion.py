import torch
import triton
import triton.language as tl

def pattern(a, b, c, d, e):
    # Match concatenation of 4 tensors followed by view operation
    concat_result = torch.cat([a, b, c, d], -1)  # tmp_2
    view_result = concat_result.view(1, -1, e.shape[0])  # tmp_3
    # We don't return concat_result because it's not observable in the final output (tmp_2 = None)
    return view_result

def replacement_args(a, b, c, d, e):
    # Pass the concatenation parameters and the layer norm norm_size
    return (a, b, c, d, e)

@triton.jit
def optimized_concat_view_kernel(
    input_ptrs_ptr,  # Pointer to array of input tensor pointers
    out_ptr,
    batch_size,
    height,
    width,
    in_channels,
    out_channels,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that concatenates 4 input tensors and reshapes them in one operation.
    This avoids the intermediate memory allocation from torch.cat + view.
    """
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    
    # Calculate which slice of output we're responsible for
    total_elements_per_batch = height * width * out_channels
    
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < total_elements_per_batch
    
    # Transform linear offset to 3D coordinates: [1, H*W, out_channels]
    spatial_size = height * width
    
    # For each offset in this block, calculate which input tensor and which position
    for i in range(4):
        input_base_ptr = tl.load(input_ptrs_ptr + i * 8)  # Assuming 8-byte pointer size
        
        # Calculate which elements belong to this input tensor
        input_offset_start = i * (spatial_size * in_channels)
        input_offset_end = (i + 1) * (spatial_size * in_channels)
        
        # Create mask for elements belonging to this input tensor within this block
        start_mask = (offsets >= input_offset_start) & (offsets < input_offset_end) & mask
        element_mask = start_mask
        
        if tl.any(element_mask):
            # Calculate position within this input tensor
            local_offset = offsets - input_offset_start
            spatial_pos = local_offset // in_channels
            channel_pos = local_offset % in_channels
            
            # Load from input tensor
            input_ptr = input_base_ptr + spatial_pos * in_channels + channel_pos
            val = tl.load(input_ptr, mask=element_mask, other=0.0)
            
            # Store to output at correct position
            out_offset = spatial_pos * out_channels + channel_pos
            tl.store(out_ptr + out_offset, val, mask=element_mask)

@torch.fx.wrap
def optimized_concat_view(a, b, c, d, norm_size):
    """
    Optimized function that concatenates 4 input tensors and reshapes to (1, N, D).
    
    Args:
        a, b, c, d: Input tensors of shape [batch, height, width, channels]
        norm_size: The target feature dimension (for shape inference)
    
    Returns:
        Reshaped tensor of shape [1, N, D] where N = batch * height * width * 4
    """
    # Get input shapes - assume all inputs have same shape [batch, height, width, channels]
    batch_size, height, width, in_channels = a.shape
    
    # Calculate output shape: 1, N, D where N is flattened spatial dimensions
    spatial_size = batch_size * height * width
    out_channels = norm_size
    
    # Create output tensor
    out = torch.empty((batch_size, spatial_size, out_channels), 
                      dtype=a.dtype, device=a.device)
    
    # If we only have 1 batch, we can optimize for the common case
    if batch_size == 1:
        # Handle the common case with direct memory copying
        total_elements = spatial_size * out_channels
        BLOCK_SIZE = 1024
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Create array of input tensor pointers
        input_ptrs = torch.tensor([a.data_ptr(), b.data_ptr(), c.data_ptr(), d.data_ptr()],
                                 dtype=torch.int64, device=a.device)
        
        # Optimized kernel for the common case
        @triton.jit
        def simple_concat_view_kernel(
            in_ptrs_ptr,
            out_ptr,
            total_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < total_elements
            
            # Calculate spatial and channel positions for each offset
            spatial_size = height * width
            spatial_pos = offsets // out_channels
            channel_pos = offsets % out_channels
            
            # For each input tensor, copy its portion to output
            for i in range(4):
                in_ptr = tl.load(in_ptrs_ptr + i * 8)  # Load input tensor pointer
                # Calculate which slice this input负责负责
                input_start = i * spatial_size
                input_end = (i + 1) * spatial_size
                
                # Create mask for elements from this input tensor
                input_mask = (spatial_pos >= input_start) & (spatial_pos < input_end) & mask
                
                if tl.any(input_mask):
                    # Calculate local spatial position
                    local_spatial = spatial_pos - input_start
                    # Calculate source offset in input tensor
                    src_offset = local_spatial * in_channels + channel_pos
                    # Load value from input
                    val = tl.load(in_ptr + src_offset, mask=input_mask, other=0.0)
                    # Store to output
                    output_offset = spatial_pos * out_channels + channel_pos
                    tl.store(out_ptr + output_offset, val, mask=input_mask)
        
        # Launch the kernel
        simple_concat_view_kernel[(num_programs,)](
            input_ptrs,
            out,
            total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # General case - just use standard PyTorch operations as fallback
        concat_result = torch.cat([a, b, c, d], -1)
        out = concat_result.view(batch_size, -1, out_channels)
    
    return out

def replacement_func():
    return optimized_concat_view