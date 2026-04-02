import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, tensor_to_concat):
    # Match the entire computation sequence:
    # conv2d -> stack -> sum -> cat (with redundant stack+sum)
    conv_result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    stacked = torch.stack([conv_result], dim=0)
    summed = stacked.sum(dim=0)
    final_result = torch.cat([summed, tensor_to_concat], 1)
    return final_result

def replacement_args(input_tensor, weight_tensor, bias_tensor, tensor_to_concat):
    return (input_tensor, weight_tensor, bias_tensor, tensor_to_concat)

@triton.jit
def simple_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel to demonstrate the pattern"""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements # Mask to ensure we don't go out of bounds
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate
    out = x + y
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_fused_operation(input_tensor, weight_tensor, bias_tensor, tensor_to_concat):
    """Simple fused operation that skips redundant stack+sum"""
    
    # First compute conv2d
    conv_result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    
    # Skip redundant stack+sum operations and directly concatenate
    result = torch.cat([conv_result, tensor_to_concat], 1)
    
    return result

def replacement_func():
    return simple_fused_operation