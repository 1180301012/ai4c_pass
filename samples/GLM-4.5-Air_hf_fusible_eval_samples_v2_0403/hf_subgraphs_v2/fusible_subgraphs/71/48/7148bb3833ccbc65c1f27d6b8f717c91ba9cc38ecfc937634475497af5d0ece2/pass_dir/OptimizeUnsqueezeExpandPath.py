import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_9 = in_1.unsqueeze(-1)
    tmp_1 = None
    tmp_10 = tmp_9.unsqueeze(-1)
    tmp_9 = None
    return tmp_10

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def expand_kernel(
    input_ptr,
    output_ptr,
    input_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < input_size
    
    # Load input data
    input_data = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Expand from [C] to [C, 1, 1] using a more efficient approach
    # Instead of complex broadcasting, we just store the data directly
    # since the expanded dimensions are just singletons
    expanded_offset = offset  # [C] offset maps to [C, 1, 1] directly
    tl.store(output_ptr + expanded_offset, input_data, mask=mask)

@torch.fx.wrap
def expand_1_1_optimized_torch(in_1):
    # Input is [C], output should be [C, 1, 1]
    input_size = in_1.shape[0]
    output_shape = (input_size, 1, 1)
    output = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Optimized threshold and block size selection
    if input_size > 512:
        # Use dynamic block size based on input size for better GPU utilization
        if input_size < 4096:
            BLOCK_SIZE = 512
        elif input_size < 16384:
            BLOCK_SIZE = 1024
        else:
            BLOCK_SIZE = 2048
        
        num_programs = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        expand_kernel[(num_programs,)](
            input_ptr=in_1,
            output_ptr=output,
            input_size=input_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # For small inputs, use native PyTorch operations which are efficient
        output = in_1.unsqueeze(-1).unsqueeze(-1)
    
    return output

def replacement_func():
    return expand_1_1_optimized_torch