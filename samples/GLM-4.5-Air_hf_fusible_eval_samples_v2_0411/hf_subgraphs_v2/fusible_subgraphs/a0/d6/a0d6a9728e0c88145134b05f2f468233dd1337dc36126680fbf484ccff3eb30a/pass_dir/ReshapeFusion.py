import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Fuse the view -> transpose -> reshape sequence
    # This pattern appears: [B, H, D] -> view(1, B, 1, D) -> transpose(1,2) -> reshape(1, 1, B*D)
    tmp_4 = input_tensor.view(1, input_tensor.shape[0], 1, input_tensor.shape[2])
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, input_tensor.shape[0] * input_tensor.shape[2])
    return tmp_6

def replacement_args(input_tensor):
    return (input_tensor,)

# Direct flat fusion for the shape transformation at module level
@torch.fx.wrap
def fused_reshape_fusion(input_tensor):
    batch_size, head_dim = input_tensor.shape[0], input_tensor.shape[2] if len(input_tensor.shape) >= 3 else input_tensor.shape[1]
    # Directly reshape to final output shape: [1, 1, batch_size * head_dim]
    return input_tensor.reshape(1, 1, batch_size * head_dim)

# For the case where input is 3D [B, H, D] at module level
@torch.fx.wrap
def fused_reshape_3d(input_tensor):
    batch_size, num_heads, head_dim = input_tensor.shape
    final_size = batch_size * head_dim  # B * D
    # Directly reshape: [B, H, D] -> [1, 1, B*D]
    return input_tensor.reshape(1, 1, final_size)

@torch.fx.wrap
def fused_reshape_2d(input_tensor):
    batch_size, head_dim = input_tensor.shape
    final_size = batch_size * head_dim
    # Directly reshape: [B, D] -> [1, 1, B*D]
    return input_tensor.reshape(1, 1, final_size)

@triton.jit
def direct_reshape_kernel(
    input_ptr, output_ptr,
    batch_size, head_dim, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data (flattened)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Store directly to output (no transformation needed between flattening and final reshape)
    tl.store(output_ptr + offsets, input_data, mask=mask)

# Optimized reshape fusion function at module level
@torch.fx.wrap
def optimized_reshape_fusion(input_tensor):
    if len(input_tensor.shape) == 3:
        batch_size, num_heads, head_dim = input_tensor.shape
        final_elements = batch_size * head_dim
    elif len(input_tensor.shape) == 2:
        batch_size, head_dim = input_tensor.shape
        final_elements = batch_size * head_dim
    else:
        # For other cases, just use standard reshape
        return input_tensor.reshape(1, 1, input_tensor.numel())
    
    # Create output tensor with final shape [1, 1, final_elements]
    output = torch.empty(1, 1, final_elements, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Flatten input and copy directly to output
    flattened_input = input_tensor.flatten()
    
    # Use Triton kernel for direct memory copy to GPU
    BLOCK_SIZE = 1024
    num_blocks = (final_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    direct_reshape_kernel[(num_blocks,)](
        flattened_input, output,
        batch_size, head_dim, final_elements,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_reshape_fusion