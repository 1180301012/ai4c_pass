import torch
import triton
import triton.language as tl

def pattern(conv3d_output):
    # Match the sequence: flatten(2) -> transpose(1, 2)
    tmp_4 = conv3d_output.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    return tmp_5

def replacement_args(conv3d_output):
    return (conv3d_output,)

@triton.jit
def optimized_flatten_transpose_kernel(
    input_ptr,
    output_ptr,
    N, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the flattened dimensions: [N*C*D, H, W] -> after transpose [N*C*W, D, H]
    input_size_per_block = H * W
    output_size_per_block = D * H
    
    # Each program handles one element in the flattened output space
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C * W)
    
    # Calculate output indices
    output_idx = offsets
    w = output_idx % W
    flat_nc = output_idx // W
    c = flat_nc % C
    n = flat_nc // C
    
    # Calculate input indices (after flatten and transpose)
    # Original shape after flatten: [N*C*D, H, W]
    # After transpose: [N*C*W, D, H]
    input_flat_idx = n * C * D + c * D + (w // H)
    input_h_idx = w % H
    
    # Load input value
    input_offset = input_flat_idx * H + input_h_idx
    input_val = tl.load(input_ptr + input_offset, mask=(input_flat_idx < (N * C * D)) & (input_h_idx < H))
    
    # Store output value
    output_offset = output_idx
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def optimized_flatten_transpose(input_tensor):
    """Optimized flatten(2) followed by transpose(1, 2)"""
    input_shape = input_tensor.shape  # Should be [1, 768, 2, 16, 16] after conv3d
    N, C, D, H, W = input_shape
    
    # Output shape after flatten(2): [N*C*D, H, W] -> [1*768*2, 16, 16] = [1536, 16, 16]
    # After transpose(1, 2): [N*C*W, D, H] -> [1*768*16, 2, 16] = [12288, 2, 16]
    output_shape = (N * C * W, D, H)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Reshape for kernel processing - we're working on the flattened data
    total_elements = N * C * W * D * H
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_flatten_transpose_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        N=N, C=C, D=D, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_flatten_transpose