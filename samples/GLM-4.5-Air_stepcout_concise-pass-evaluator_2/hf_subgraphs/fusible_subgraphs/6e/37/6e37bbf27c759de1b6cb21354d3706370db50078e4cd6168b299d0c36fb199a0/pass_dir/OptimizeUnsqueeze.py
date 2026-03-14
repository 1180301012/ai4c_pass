import torch
import triton
import triton.language as tl

def pattern(tensor, dim):
    result = tensor.unsqueeze(dim)
    return result

def replacement_args(tensor, dim):
    return (tensor, dim)

@triton.jit
def unsqueeze_kernel(
    input_ptr,
    output_ptr,
    input_size_0: tl.constexpr,
    input_size_1: tl.constexpr,
    input_size_2: tl.constexpr,
    output_size_0: tl.constexpr,
    output_size_1: tl.constexpr,
    output_size_2: tl.constexpr,
    output_size_3: tl.constexpr,
    stride_inp_0: tl.constexpr,
    stride_inp_1: tl.constexpr,
    stride_inp_2: tl.constexpr,
    stride_out_0: tl.constexpr,
    stride_out_1: tl.constexpr,
    stride_out_2: tl.constexpr,
    stride_out_3: tl.constexpr,
    BLOCK_SIZE_0: tl.constexpr,
    BLOCK_SIZE_1: tl.constexpr,
    BLOCK_SIZE_2: tl.constexpr,
):
    # Get program ID grid
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_2 = tl.program_id(2)
    
    # Create offsets for dimensions
    off_0 = pid_0 * BLOCK_SIZE_0 + tl.arange(0, BLOCK_SIZE_0)
    off_1 = pid_1 * BLOCK_SIZE_1 + tl.arange(0, BLOCK_SIZE_1)
    off_2 = pid_2 * BLOCK_SIZE_2 + tl.arange(0, BLOCK_SIZE_2)
    
    # Create masks for bounds checking
    mask_0 = off_0 < output_size_0
    mask_1 = off_1 < output_size_1
    mask_2 = off_2 < output_size_2
    
    # Calculate input coordinates based on unsqueeze dimension
    # If unsqueeze(dim=1), then input has shape [..., D2, D3] and output has [..., 1, D2, D3]
    input_off_0 = off_0  # Dimension 0 stays the same
    input_off_1 = off_2  # Dimension 1 in output corresponds to dimension 1-1=0 in input
    input_off_2 = (off_1 * input_size_2 + off_2) // (dim + 1)  # Need to handle the unsqueeze carefully
    
    # For unsqueeze(dim=1), input [D0, D1, D2] -> output [D0, D1, 1, D2]
    # So output [d0, d1, d2, d3] maps to input [d0, d1*output_size_3 + d3, :]
    
    # For our case: input [1, 128, 128] -> unsqueeze(1) -> [1, 1, 128, 128]
    # So output [batch, unsqueeze_dim, seq, hidden] maps to input [batch, seq*hidden + hidden, :]
    
    input_batch = off_0
    input_flat = off_1 * output_size_3 + off_2  # Combine dimensions that were split
    
    # Create input mask
    input_mask = mask_0 & (input_flat < input_size_1 * input_size_2)
    
    # Load input data
    input_offset = input_batch * stride_inp_0 + input_flat * stride_inp_1
    input_data = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)
    
    # Store output data
    output_offset = (
        off_0 * stride_out_0 + 
        off_1 * stride_out_1 + 
        off_2 * stride_out_2 + 
        off_2 * stride_out_3  # This handles the singleton dimension properly
    )
    
    # Create output mask for all dimensions
    output_mask = mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :]
    
    # Store the same data to all positions in the expanded dimension
    tl.store(output_ptr + output_offset, input_data, mask=output_mask)

@torch.fx.wrap  
def triton_unsqueeze_optimized(tensor, dim):
    # Get input shape
    input_shape = tensor.shape
    
    # Calculate output shape
    output_shape = list(input_shape)
    output_shape.insert(dim, 1)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Triton kernel parameters
    BLOCK_SIZE_0 = 64   # Batch dimension
    BLOCK_SIZE_1 = 32   # Dim to expand (will become multiple iterations)
    BLOCK_SIZE_2 = 32   # Remaining dimensions
    
    # Get input strides
    stride_inp_0 = tensor.stride(0) if len(input_shape) > 0 else 0
    stride_inp_1 = tensor.stride(1) if len(input_shape) > 1 else 0
    stride_inp_2 = tensor.stride(2) if len(input_shape) > 2 else 0
    
    # Calculate output strides (add singleton dimension)
    output_strides = []
    current_stride = 1
    for i, size in enumerate(output_shape):
        output_strides.insert(0, current_stride)
        current_stride *= size
    
    stride_out_0 = output_strides[0]
    stride_out_1 = output_strides[1] 
    stride_out_2 = output_strides[2]
    stride_out_3 = output_strides[3]
    
    # Calculate grid sizes
    grid_0 = (output_shape[0] + BLOCK_SIZE_0 - 1) // BLOCK_SIZE_0
    grid_1 = (output_shape[dim + 1] + BLOCK_SIZE_1 - 1) // BLOCK_SIZE_1  # This is tricky as we inserted a dim
    grid_2 = (output_shape[3] + BLOCK_SIZE_2 - 1) // BLOCK_SIZE_2
    
    # For simplicity, let's handle 3D input -> 4D output case specifically
    if len(input_shape) == 3 and dim == 1:
        input_size_0, input_size_1, input_size_2 = input_shape
        output_size_0, output_size_1, output_size_2, output_size_3 = output_shape
        
        # Launch kernel with optimized parameters for this specific case
        unsqueeze_kernel[(grid_0, grid_1, grid_2)](
            tensor,
            output,
            input_size_0, input_size_1, input_size_2,
            output_size_0, output_size_1, output_size_2, output_size_3,
            stride_inp_0, stride_inp_1, stride_inp_2,
            stride_out_0, stride_out_1, stride_out_2, stride_out_3,
            BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2
        )
    else:
        # Fall back to regular PyTorch for unsupported cases
        return tensor.unsqueeze(dim)
    
    return output

def replacement_func():
    return triton_unsqueeze_optimized