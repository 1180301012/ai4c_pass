import torch
import triton
import triton.language as tl

def input_tensor_4():
    # This matches tmp_4 input directly for the following operations
    pass

def pattern(input_tensor):
    # Matches: slice -> reshape(1, 27, 27, -1) -> permute(0, 3, 1, 2)
    tmp_11 = input_tensor[slice(None, 729, None)]
    tmp_12 = tmp_11.reshape(1, 27, 27, -1)
    tmp_13 = tmp_12.permute(0, 3, 1, 2)
    return tmp_13

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def fused_slice_reshape_permute_kernel(
    input_ptr,
    output_ptr,
    output_height,
    output_width,
    features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = output_height * output_width * features
    
    # Calculate offsets for parallel processing
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    h = offsets // (output_width * features)
    rem = offsets % (output_width * features)
    w = rem // features
    f = rem % features
    
    # Load from input: slice and map to [h, w, f]
    input_offset = h * output_width * features + w * features + f
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store to output: [batch=1, features, h, w] format
    output_offset = 1 * features * output_height * output_width + f * output_height * output_width + h * output_width + w
    tl.store(output_ptr + output_offset, x, mask=mask)

@torch.fx.wrap
def fused_slice_reshape_permute_gpu(input_tensor):
    # Input: [732, 12] - take first 729 elements
    # Output should be [1, 1, 27, 27] based on the operations
    slice_size = 729
    
    # Extract slice and determine final dimensions
    input_slice = input_tensor[:slice_size]
    
    # Final output shape: [1, 1, 27, 27]
    batch_size = 1
    output_channels = 1
    output_height = 27
    output_width = 27
    
    output = torch.empty(batch_size, output_channels, output_height, output_width, 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    N = output_height * output_width * output_channels
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_slice_reshape_permute_kernel[(num_programs,)](
        input_ptr=input_slice,
        output_ptr=output,
        output_height=output_height,
        output_width=output_width,
        features=output_channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_slice_reshape_permute_gpu