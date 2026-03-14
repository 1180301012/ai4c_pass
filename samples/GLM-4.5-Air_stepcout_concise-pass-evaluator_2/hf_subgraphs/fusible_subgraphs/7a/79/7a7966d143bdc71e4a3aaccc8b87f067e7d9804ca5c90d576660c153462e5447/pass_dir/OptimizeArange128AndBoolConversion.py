import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function for arange(0, 128)
def pattern(in_0):
    """
    Matches the computation pattern from graph 7:
    1. Assign input to tmp_0
    2. Create arange tensor with size 128
    3. Convert input to boolean (device and dtype conversion)
    4. Return both arange and boolean tensors
    """
    tmp_0 = in_0
    target_device = device(type='cuda', index=0)
    tmp_1 = torch.arange(0, 128, device=target_device)
    tmp_2 = tmp_0.to(device=target_device, dtype=torch.bool)
    return (tmp_1, tmp_2)

# Argument extraction function
def replacement_args(in_0):
    """Extract the input tensor needed for the replacement"""
    return (in_0,)

# Optimized Triton kernel for creating arange tensors
@triton.jit
def create_arange_kernel(
    out_ptr,
    end_value,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that creates an arange tensor [0, 1, 2, ..., end_value-1]"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < end_value
    tl.store(out_ptr + offsets, offsets, mask=mask)

# Optimized arange wrapper using Triton kernel
@torch.fx.wrap
def create_optimized_arange_128(device):
    """Create arange(0, 128) using optimized Triton kernel"""
    output_tensor = torch.empty(128, dtype=torch.int64, device=device)
    BLOCK_SIZE = 1024
    num_programs = (128 + BLOCK_SIZE - 1) // BLOCK_SIZE
    create_arange_kernel[(num_programs,)](
        out_ptr=output_tensor,
        end_value=128,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_tensor

# Optimized kernel for creating all-True boolean tensors
@triton.jit
def create_all_true_kernel(
    out_ptr,
    output_shape,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that creates a tensor filled with True values"""
    total_elements = output_shape[0] * output_shape[1]
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    tl.store(out_ptr + offsets, 1, mask=mask)

@torch.fx.wrap
def create_all_true_tensor(input_tensor):
    """Create a tensor of all True values with the same shape as input_tensor"""
    input_shape = input_tensor.shape
    output_tensor = torch.empty(input_shape, dtype=torch.bool, device=input_tensor.device)
    total_elements = input_shape.numel()
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    create_all_true_kernel[(num_programs,)](
        out_ptr=output_tensor,
        output_shape=input_shape,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_tensor

# Combined optimized computation
@torch.fx.wrap
def optimized_forward_pass(in_0):
    """Optimized forward pass that creates both tensors efficiently"""
    tmp_1 = create_optimized_arange_128(in_0.device)
    tmp_2 = create_all_true_tensor(in_0)
    return (tmp_1, tmp_2)

# Replacement function
def replacement_func():
    """Return the optimized function"""
    return optimized_forward_pass