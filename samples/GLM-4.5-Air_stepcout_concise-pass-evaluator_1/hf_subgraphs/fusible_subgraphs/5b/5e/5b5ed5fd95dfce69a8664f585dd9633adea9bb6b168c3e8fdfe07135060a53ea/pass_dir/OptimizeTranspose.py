import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    return input_tensor.transpose(1, 2)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_transpose_kernel(
    input_ptr,
    output_ptr,
    N, C, H,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized transpose for tensors where we swap dims 1 and 2
    # Input shape: [N, C, H], output shape: [N, H, C]
    pid = tl.program_id(0)
    num_programs = tl.cdiv(N * C * H, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
        
    # Compute offsets
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H
    
    # Calculate indices for input [N, C, H]
    # offset = n*C*H + c*H + h
    h_idx = offsets % H
    remainder = offsets // H
    c_idx = remainder % C
    n_idx = remainder // C
    
    # Calculate output offset for [N, H, C]
    # output_offset = n*H*C + h*C + c
    output_offset = n_idx * H * C + h_idx * C + c_idx
    
    # Load input and store to output with transposed layout
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def optimized_transpose(input_tensor):
    """Optimized transpose using Triton kernel"""
    if len(input_tensor.shape) != 3:
        # Fall back to regular transpose for non-3D tensors
        return input_tensor.transpose(1, 2)
    
    N, C, H = input_tensor.shape
    total_elements = N * C * H
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output with transposed shape [N, H, C]
    output = torch.empty((N, H, C), dtype=input_tensor.dtype, device=input_tensor.device)
    
    optimized_transpose_kernel[(num_programs,)](
        input_tensor, output, N, C, H, BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_transpose