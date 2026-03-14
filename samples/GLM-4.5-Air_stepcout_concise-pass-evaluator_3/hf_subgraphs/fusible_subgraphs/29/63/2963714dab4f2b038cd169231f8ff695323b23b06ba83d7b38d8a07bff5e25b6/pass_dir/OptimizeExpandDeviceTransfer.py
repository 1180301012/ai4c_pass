import torch
from torch import device
import triton
import triton.language as tl

def pattern(input_tensor):
    # Pattern matching: unsqueeze + expand + device transfer
    tmp_5 = input_tensor.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    tmp_7 = tmp_6.to(device(type='cuda', index=0))
    return tmp_7

@triton.jit
def optimized_expand_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element
    expand_idx = tl.program_id(0)  # 0, 1, or 2 for the three expansions
    batch_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Calculate input offset (expansion doesn't change the underlying data)
    input_offset = batch_idx * seq_len + seq_idx
    
    # Calculate output offset (3 * batch_size * seq_len layout)
    output_offset = expand_idx * batch_size * seq_len + batch_idx * seq_len + seq_idx
    
    # Load input value
    val = tl.load(input_ptr + input_offset)
    
    # Store expanded output value (same value replicated 3 times)
    tl.store(output_ptr + output_offset, val)

@torch.fx.wrap
def optimized_expand_device_transfer(input_tensor):
    batch_size, seq_len = input_tensor.shape
    
    # Create output tensor directly on GPU with expanded shape
    output = torch.empty((3, batch_size, seq_len), dtype=input_tensor.dtype, device='cuda')
    
    # Calculate grid dimensions
    grid = (3, batch_size, seq_len)
    
    # Launch optimized kernel that creates expanded tensor directly on GPU
    optimized_expand_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=1,
    )
    
    return output

def replacement_args(input_tensor):
    return (input_tensor,)

def replacement_func():
    return optimized_expand_device_transfer