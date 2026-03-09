import torch
import triton
import triton.language as tl

def pattern(scaled_input, in_0):
    # Match the add + ReLU operation sequence
    tmp_3 = scaled_input + in_0
    tmp_4 = tmp_3
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    return tmp_5

def replacement_args(scaled_input, in_0):
    return (scaled_input, in_0)

@triton.jit
def fused_add_relu_kernel(
    scaled_input_ptr,
    in_0_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    scaled_input = tl.load(scaled_input_ptr + offsets, mask=mask, other=0.0)
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: add + ReLU
    added = scaled_input + in_0
    out = tl.maximum(added, 0.0)
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_relu_operation(scaled_input, in_0):
    N = scaled_input.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(scaled_input)
    
    fused_add_relu_kernel[(num_programs,)](
        scaled_input_ptr=scaled_input,
        in_0_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_relu_operation