import torch
import triton
import triton.language as tl
from torch import device

def pattern(tmp_2, tmp_4):
    # Pattern: detach() + type_as() sequence (simplified)
    tmp_6 = tmp_2.detach()
    tmp_7 = tmp_6.type_as(tmp_4)
    return tmp_7

def replacement_args(tmp_2, tmp_4):
    return (tmp_2, tmp_4)

@triton.jit
def fused_conversion_kernel(
    input_ptr,
    target_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor (with det semantic - ensure we don't track gradients)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load target tensor for type information
    target_val = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Perform type conversion (using target tensor's dtype)
    if target_val.dtype == tl.float16:
        output_val = input_val.to(tl.float16)
    elif target_val.dtype == tl.bfloat16:
        output_val = input_val.to(tl.bfloat16)
    else:
        output_val = input_val.to(tl.float32)
    
    # Store result (implicitly copies to device if needed)
    tl.store(output_ptr + offsets, output_val, mask=mask)

@torch.fx.wrap
def fused_device_type_conversion(source_tensor, target_tensor):
    # Determine which tensor is on CPU and which provides target dtype
    if source_tensor.device.type == 'cpu' and target_tensor.device.type == 'cuda':
        cpu_tensor = source_tensor
        cuda_tensor = target_tensor
    elif target_tensor.device.type == 'cpu' and source_tensor.device.type == 'cuda':
        cpu_tensor = target_tensor
        cuda_tensor = source_tensor
    else:
        # Both on same device, just do type conversion
        return source_tensor.type_as(target_tensor)
    
    N = cpu_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor on correct device with correct dtype
    output = torch.empty(N, dtype=cuda_tensor.dtype, device=cuda_tensor.device)
    
    fused_conversion_kernel[(num_programs,)](
        input_ptr=cpu_tensor,
        target_ptr=cuda_tensor,
        output_ptr=output,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to match expected output shape
    return output.reshape(source_tensor.shape)

def replacement_func():
    return fused_device_type_conversion