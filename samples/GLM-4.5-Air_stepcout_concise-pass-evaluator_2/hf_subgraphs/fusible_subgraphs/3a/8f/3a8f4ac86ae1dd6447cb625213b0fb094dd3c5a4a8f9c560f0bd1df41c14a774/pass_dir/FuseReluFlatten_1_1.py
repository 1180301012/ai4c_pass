import torch
import triton
import triton.language as tl

# Pattern matching function - matches ReLU + Flatten
def pattern(input_tensor):
    # Pattern matches ReLU followed by flatten(1, -1)
    relu_output = torch.nn.functional.relu(input_tensor, inplace=False)
    flatten_output = relu_output.flatten(1, -1)
    return flatten_output

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized kernel - fused ReLU + Flatten
@triton.jit
def fused_relu_flatten_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + Flatten kernel
    Input shape: [batch_size, channels, 1, 1]
    Output shape: [batch_size, channels]
    """
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    total_elements = batch_size * channels
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Translate 1D offset to 4D coordinates [B, C, 1, 1]
    # Since last two dimensions are 1x1, they don't contribute to offset
    b = offsets // channels
    c = offsets % channels
    
    # Calculate 4D offset for input tensor [B, C, 1, 1]
    input_offset = b * channels * 1 * 1 + c * 1 * 1 + 0 * 1 + 0
    
    # Load input element
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Apply ReLU
    relu_out = tl.maximum(x, 0.0)
    
    # Store output in flattened layout
    tl.store(output_ptr + offsets, relu_out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(input_tensor):
    """
    Triton kernel that fuses ReLU activation with flattening
    Optimized for input tensors with last two dimensions = 1x1
    """
    batch_size, channels, h, w = input_tensor.shape
    
    # Since h=w=1 for our specific case, flatten(1, -1) is straightforward
    assert h == 1 and w == 1, "This fusion only works for 1x1 spatial dimensions"
    
    total_elements = batch_size * channels
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty((batch_size, channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    fused_relu_flatten_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fused_relu_flatten