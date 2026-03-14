import torch
import triton
import triton.language as tl

@triton.jit
def fused_normalization_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    pid = tl.program_id(0)
    batch_size = tl.num_programs(0)
    
    # Calculate indices for this batch element
    batch_idx = pid
    
    # Load entire batch element from shared memory
    offsets = batch_idx * hidden_size + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean using Triton operations
    sum_x = tl.sum(x)
    mean_x = sum_x / hidden_size
    
    # Compute variance (E[x^2] - E[x]^2)
    x2 = x * x
    sum_x2 = tl.sum(x2)
    var_x = sum_x2 / hidden_size - mean_x * mean_x
    
    # Add epsilon and compute rsqrt
    eps = 1e-06
    rsqrt_var = tl.rsqrt(var_x + eps)
    
    # Normalize: (x - mean) * rsqrt_var
    normalized = (x - mean_x) * rsqrt_var
    
    # Store result
    tl.store(output_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap
def fused_normalization(input_tensor):
    batch_size, seq_len, hidden_size = input_tensor.shape
    n_elements = input_tensor.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    fused_normalization_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(tmp_1):
    # Pattern to match the normalization subgraph
    tmp_4 = tmp_1.to(torch.float32)
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_1 * tmp_8
    return tmp_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9

def replacement_args(tmp_1):
    return (tmp_1,)

def replacement_func():
    return fused_normalization