import torch
import triton
import triton.language as tl

# Pattern matching function - matches sigmoid -> view -> multiply sequence
def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel optimized for specific tensor shapes [1,512] and [1,512,64,64]
@triton.jit
def tensor_shape_optimized_kernel(
    input_0_ptr,    # [1, 512] - sigmoid input
    input_1_ptr,    # [1, 512, 64, 64] - broadcast multiply input
    output_ptr,     # [1, 512, 64, 64] - result
    n_channels,     # 512
    height,         # 64  
    width,          # 64
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for 1D grid
    pid = tl.program_id(0)
    
    # Calculate offset for this program
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total elements in the output tensor
    total_elements = n_channels * height * width
    mask = offset < total_elements
    
    # Get valid offsets 
    valid_offset = offset[mask]
    
    # Calculate channel, row, and column indices
    # This division layout is optimized for [512, 64, 64] structure
    channel_idx = valid_offset // (height * width)
    remaining = valid_offset % (height * width)
    row_idx = remaining // width
    col_idx = remaining % width
    
    # Calculate memory offsets for efficient access
    # Channel-major layout for best memory locality
    channel_offset = channel_idx * (height * width)
    input_1_offset = valid_offset
    
    # Load input data with coalesced memory access pattern
    input_1_val = tl.load(input_1_ptr + input_1_offset, mask=mask)
    
    # Load and compute sigmoid values efficiently
    sigmoid_input = tl.load(input_0_ptr + channel_idx, mask=mask)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-sigmoid_input.to(tl.float32))).to(input_1_val.dtype)
    
    # Perform fused multiplication with broadcasting
    result = input_1_val * sigmoid_val
    
    # Store result with optimized memory pattern
    tl.store(output_ptr + valid_offset, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def tensor_shape_optimized(in_0, in_1):
    # Extract tensor dimensions for specific optimization
    batch_size, n_channels = in_0.shape
    _, _, height, width = in_1.shape
    
    # Calculate total elements in 4D tensor
    total_elements = n_channels * height * width
    
    # Create output tensor with same properties
    output = torch.empty_like(in_1)
    
    # Block size optimized for [1,512,64,64] tensor shape
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with shape-specific parameters
    tensor_shape_optimized_kernel[(num_programs,)](
        in_0,
        in_1,
        output,
        n_channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return tensor_shape_optimized