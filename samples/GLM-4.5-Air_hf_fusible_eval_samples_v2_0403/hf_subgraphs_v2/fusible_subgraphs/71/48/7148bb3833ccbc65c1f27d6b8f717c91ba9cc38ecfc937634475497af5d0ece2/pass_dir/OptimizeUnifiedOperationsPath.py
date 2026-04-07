import torch
import triton
import triton.language as tl

def pattern(in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    return tmp_2

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute ReLU: max(x, 0)
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def relu_optimized_torch(x):
    n_elements = x.numel()
    
    # Use optimized kernel for medium to large tensors
    if n_elements > 2048:
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        output = torch.empty_like(x)
        
        relu_kernel[(num_programs,)](
            input_ptr=x,
            output_ptr=output,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
    else:
        # For small tensors, PyTorch's native ReLU is highly optimized
        return torch.nn.functional.relu(x, inplace=False)

def replacement_func():
    return relu_optimized_torch