import torch
import triton
import triton.language as tl

@triton.jit
def fused_relu_view_unsqueeze_kernel(
    input_ptr,
    relu_output_ptr,
    final_output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    spatial_size: tl.constexpr,
    final_spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    n_elements = batch_size * channels * spatial_size
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU operation
    relu_out = tl.maximum(x, 0.0)
    
    # Store ReLU result
    tl.store(relu_output_ptr + offsets, relu_out, mask=mask)
    
    # Calculate final output indices for reshaped data
    # We need to map from flattened linear offsets to [batch, 1, channels, final_spatial_size]
    elem_per_final_tensor = channels * final_spatial_size
    elem_per_batch = elem_per_final_tensor
    
    final_offsets = []
    for offset in offsets:
        if offset < n_elements:
            batch = offset // (channels * spatial_size)
            remainder = offset % (channels * spatial_size)
            channel = remainder // spatial_size
            spatial = remainder % spatial_size
            
            # Map to final shape: [batch, 1, channels, final_spatial_size]
            final_offset = batch * elem_per_batch + 0 * elem_per_final_tensor + channel * final_spatial_size + spatial
            final_offsets.append(final_offset)
        else:
            final_offsets.append(0)
    
    final_offsets = tl.tensor(final_offsets, dtype=tl.int64)
    
    # Store final reshaped result
    tl.store(final_output_ptr + final_offsets, relu_out, mask=mask)

@torch.fx.wrap
def fused_relu_view_unsqueeze(input_tensor):
    # Get input dimensions
    batch_size, channels, height, width = input_tensor.shape
    spatial_size = height * width
    final_spatial_size = 4096  # This matches the target shape from view(., 4096)
    
    # Compute total elements
    total_elements = batch_size * channels * spatial_size
    
    # Output tensors
    relu_output = torch.empty_like(input_tensor)
    final_output = torch.empty(batch_size, 1, channels, final_spatial_size, device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Block size configuration
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_relu_view_unsqueeze_kernel[(num_programs,)](
        input_ptr=input_tensor,
        relu_output_ptr=relu_output,
        final_output_ptr=final_output,
        batch_size=batch_size,
        channels=channels,
        spatial_size=spatial_size,
        final_spatial_size=final_spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return final_output, relu_output

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    # Get the batch size from input to handle both [1, 512, 64, 64] and [32, 512, 64, 64]
    batch_size = in_0.shape[0]
    tmp_1 = tmp_0.view(batch_size, 512, 4096)
    tmp_2 = tmp_1.unsqueeze(1)
    return tmp_2, tmp_0

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return fused_relu_view_unsqueeze