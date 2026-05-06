import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp1 = in_0.ne(1)
    tmp2 = tmp1.int()
    tmp3 = torch.cumsum(tmp2, dim=1)
    tmp4 = tmp3.type_as(tmp2)
    tmp5 = tmp4 + 0
    tmp6 = tmp5 * tmp2
    tmp7 = tmp6.long()
    tmp8 = tmp7 + 1
    return tmp8

def replacement_args(in_0):
    return (in_0,)



@triton.jit
def optimized_cumsum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for ne_cumsum pattern
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load input values with mask
    input_vals = tl.load(input_ptr + start + offsets, mask=offsets < n_elements, other=0)
    mask = (input_vals != 1)
    
    # Initialize cumulative counts
    cumulative = tl.zeros(BLOCK_SIZE, dtype=tl.int32)
    
    # Compute cumulative sum per sequence
    for i in range(BLOCK_SIZE):
        if offsets[i] < n_elements:
            if mask[i]:
                cumulative[i] += 1
    
    # Store result: (cumulative + 1)
    tl.store(output_ptr + start + offsets, cumulative + 1)

@torch.fx.wrap
def kernel_wrapper(input_tensor):
    N = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty(N, dtype=torch.int64, device=input_tensor.device)
    
    optimized_cumsum_kernel[(num_blocks,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return kernel_wrapper