import torch
import triton
import triton.language as tl

def pattern(tmp_0, tmp_16):
    tmp_17 = tmp_0 * tmp_16
    return tmp_17

@triton.jit
def optimized_mul_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offset < (batch_size * seq_len * hidden_dim)
    
    # Load the weight (broadcasted across batch and sequence)
    weight = tl.load(weight_ptr + offset % hidden_dim, mask=offset % hidden_dim < hidden_dim, other=0.0)
    
    # Load the input
    input_val = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Perform multiplication
    output = weight * input_val
    
    # Store result
    tl.store(output_ptr + offset, output, mask=mask)

@torch.fx.wrap
def optimized_final_mul(tmp_0, tmp_16):
    # tmp_0: [hidden_dim] - the weight
    # tmp_16: [batch_size, seq_len, hidden_dim] - the normalized input
    batch_size, seq_len, hidden_dim = tmp_16.shape
    
    output = torch.empty((batch_size, seq_len, hidden_dim), dtype=torch.bfloat16, device=tmp_0.device)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * seq_len * hidden_dim
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_mul_kernel[(num_programs,)](
        tmp_0,
        tmp_16,
        output,
        batch_size,
        seq_len,
        hidden_dim,
        BLOCK_SIZE
    )
    
    return output

def replacement_args(tmp_0, tmp_16):
    return (tmp_0, tmp_16)

def replacement_func():
    return optimized_final_mul