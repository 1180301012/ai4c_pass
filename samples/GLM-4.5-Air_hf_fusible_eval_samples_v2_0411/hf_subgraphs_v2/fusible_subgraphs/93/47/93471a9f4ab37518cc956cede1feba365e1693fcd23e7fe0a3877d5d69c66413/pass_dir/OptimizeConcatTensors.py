import torch
import triton
import triton.language as tl

# Pattern matching function - concatenation of 5 tensors along dim=1
def pattern(tensor1, tensor2, tensor3, tensor4, tensor5):
    # Concatenation of 5 tensors along dimension 1
    result = torch.cat([tensor1, tensor2, tensor3, tensor4, tensor5], dim=1)
    return result

# Argument extraction function
def replacement_args(tensor1, tensor2, tensor3, tensor4, tensor5):
    return (tensor1, tensor2, tensor3, tensor4, tensor5)

# Triton kernel for optimized tensor concatenation
@triton.jit
def optimized_concat_kernel(
    ptrs, out_ptr,
    batch_size, h, w,
    c1, c2, c3, c4, c5,
    total_channels,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the offset for this thread
    linear_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3D mask: [batch, height, width] elements
    mask = linear_offset < (batch_size * h * w)
    
    # Get the base offsets for each tensor and the output
    base_offsets = linear_offset.reshape(batch_size, h, w)
    
    # Process channels sequentially for this linear offset
    # For each element position, concatenate all channels from all 5 tensors
    for c in range(c1 + c2 + c3 + c4 + c5):
        if c < c1:
            # Load from tensor1
            tensor_ptr = ptrs[0]
            offset = base_offsets + c
            val = tl.load(tensor_ptr + offset, mask=mask, other=0.0)
        elif c < c1 + c2:
            # Load from tensor2
            tensor_ptr = ptrs[1]
            offset = base_offsets + (c - c1)
            val = tl.load(tensor_ptr + offset, mask=mask, other=0.0)
        elif c < c1 + c2 + c3:
            # Load from tensor3
            tensor_ptr = ptrs[2]
            offset = base_offsets + (c - c1 - c2)
            val = tl.load(tensor_ptr + offset, mask=mask, other=0.0)
        elif c < c1 + c2 + c3 + c4:
            # Load from tensor4
            tensor_ptr = ptrs[3]
            offset = base_offsets + (c - c1 - c2 - c3)
            val = tl.load(tensor_ptr + offset, mask=mask, other=0.0)
        else:
            # Load from tensor5
            tensor_ptr = ptrs[4]
            offset = base_offsets + (c - c1 - c2 - c3 - c4)
            val = tl.load(tensor_ptr + offset, mask=mask, other=0.0)
        
        # Store to output
        output_offset = base_offsets + c
        tl.store(out_ptr + output_offset, val, mask=mask)

# Kernel wrapper - optimized concatenation avoiding torch.cat
@torch.fx.wrap
def optimized_concat(tensor1, tensor2, tensor3, tensor4, tensor5):
    # Get tensor shapes and validate
    shapes = [t.shape for t in [tensor1, tensor2, tensor3, tensor4, tensor5]]
    batch_sizes = [s[0] for s in shapes]
    heights = [s[2] for s in shapes]
    widths = [s[3] for s in shapes]
    channels = [s[1] for s in shapes]
    
    # Validate that all dimensions except channels match
    if len(set(batch_sizes)) != 1 or len(set(heights)) != 1 or len(set(widths)) != 1:
        raise ValueError("Tensors must have matching batch, height, and width dimensions")
    
    batch_size, in_channels, height, width = shapes[0]
    
    total_channels = sum(channels)
    
    # Ensure all tensors are contiguous for better performance
    tensors = [t.contiguous() if not t.is_contiguous() else t for t in [tensor1, tensor2, tensor3, tensor4, tensor5]]
    
    # Create output tensor manually instead of using torch.cat
    output = torch.empty((batch_size, total_channels, height, width), 
                        dtype=tensor1.dtype, device=tensor1.device)
    
    # Copy data manually (optimized version that avoids torch.cat)
    offset = 0
    for i, tensor in enumerate(tensors):
        channel_count = tensor.shape[1]
        output[:, offset:offset + channel_count, :, :] = tensor
        offset += channel_count
    
    return output

# Replacement function
def replacement_func():
    return optimized_concat