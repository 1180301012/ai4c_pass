import torch
import triton
import triton.language as tl

def pattern(type_tensor, input_tensor, split_tensor):
    """
    Pattern that matches the final operations:
    tmp_8 = type_tensor.type_as(in_7)
    tmp_9 = input_tensor[:, :, :slice_idx, :]
    tmp_10 = input_tensor[:, :, slice_idx:, :]
    tmp_11 = split_tensor.tensor_split(2, -1)
    """
    # Original operations
    tmp_8 = type_tensor
    tmp_9 = input_tensor[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
    tmp_10 = input_tensor[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_11 = split_tensor.tensor_split(2, -1)
    tmp_12 = tmp_11[0]
    tmp_13 = tmp_11[1]
    
    return (tmp_13, tmp_9, tmp_10, tmp_8, tmp_12)

@triton.jit
def optimized_slice_kernel(
    input_ptr,
    out1_ptr,  # tmp_9 - first slice
    out2_ptr,  # tmp_10 - second slice  
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for slicing a tensor along the third dimension
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Get shapes from the tensor pointers
    input_shape = tl.device_shape(input_ptr)
    # For output shapes, we can compute them based on input shape
    out1_shape = (input_shape[0], input_shape[1], 1, input_shape[3])  # First part: only w=0
    out2_shape = (input_shape[0], input_shape[1], input_shape[2] - 1, input_shape[3])  # Second part: w=1 to end
    
    # Calculate total elements and work per program
    total_elements = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Calculate start and end indices for this program
    start_idx = pid * elements_per_program
    end_idx = min(start_idx + elements_per_program, total_elements)
    
    if start_idx >= total_elements:
        return
    
    # Helper function to get multi-dimensional indices
    def get_indices(idx, shape):
        # Unflatten 1D index to 4D indices
        total = idx
        f = total // (shape[1] * shape[2] * shape[3])
        total = total % (shape[1] * shape[2] * shape[3])
        h = total // (shape[2] * shape[3])
        total = total % (shape[2] * shape[3])
        w = total // shape[3]
        c = total % shape[3]
        return f, h, w, c
    
    # Process elements in this program
    for idx in range(start_idx, end_idx):
        f, h, w, c = get_indices(idx, input_shape)
        
        input_offset = f * input_shape[1] * input_shape[2] * input_shape[3] + \
                       h * input_shape[2] * input_shape[3] + \
                       w * input_shape[3] + c
        input_val = tl.load(input_ptr + input_offset, mask=(f < input_shape[0] and h < input_shape[1] and w < input_shape[2] and c < input_shape[3]), other=0.0)
        
        # Split along third dimension at w=1
        if w < 1:  # First part: slice(None, 1, None)
            out1_offset = f * out1_shape[1] * out1_shape[2] * out1_shape[3] + \
                          h * out1_shape[2] * out1_shape[3] + \
                          w * out1_shape[3] + c
            tl.store(out1_ptr + out1_offset, input_val)
        else:  # Second part: slice(1, None, None)
            out2_offset = (f - 0) * out2_shape[1] * out2_shape[2] * out2_shape[3] + \
                          (h - 0) * out2_shape[2] * out2_shape[3] + \
                          (w - 1) * out2_shape[3] + c
            tl.store(out2_ptr + out2_offset, input_val)

@torch.fx.wrap
def optimized_slicing(type_tensor, input_tensor, split_tensor):
    """Wrapper for the optimized slicing operations"""
    # Validate input shapes
    if len(input_tensor.shape) != 4:
        # Fall back to original computation if shapes are unexpected
        tmp_8 = type_tensor
        tmp_9 = input_tensor[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_10 = input_tensor[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_11 = split_tensor.tensor_split(2, -1)
        return (tmp_11[1], tmp_9, tmp_10, tmp_8, tmp_11[0])
    
    # Get tensor split results (this operation is complex to optimize in Triton)
    tmp_11 = split_tensor.tensor_split(2, -1)
    split_part_0 = tmp_11[0]
    split_part_1 = tmp_11[1]
    
    # Get input tensor shape
    input_shape = input_tensor.shape
    
    # Calculate output shapes for slicing
    out1_shape = (input_shape[0], input_shape[1], 1, input_shape[3])  # First part: only w=0
    out2_shape = (input_shape[0], input_shape[1], input_shape[2] - 1, input_shape[3])  # Second part: w=1 to end
    
    # Create output tensors
    out1 = torch.empty(out1_shape, dtype=torch.float32, device=input_tensor.device)
    out2 = torch.empty(out2_shape, dtype=torch.float32, device=input_tensor.device)
    
    # Launch Triton kernel for slicing optimization
    BLOCK_SIZE = 1024
    total_elements = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_slice_kernel[(num_programs,)](
        input_tensor,
        out1,
        out2,
        BLOCK_SIZE,
    )
    
    return (split_part_1, out1, out2, type_tensor, split_part_0)

def replacement_args(type_tensor, input_tensor, split_tensor):
    return (type_tensor, input_tensor, split_tensor)

def replacement_func():
    return optimized_slicing