import torch
import triton
import triton.language as tl

def pattern(in_2, tmp_1, tmp_0):
    """Optimize torch.nn.functional.linear operation"""
    tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    return tmp_2

def replacement_args(in_2, tmp_1, tmp_0):
    return (in_2, tmp_1, tmp_0)

@triton.jit
def linear_kernel(
    x_ptr,      # [batch_size, input_dim]
    weight_ptr, # [output_dim, input_dim] 
    bias_ptr,   # [output_dim]
    out_ptr,    # [batch_size, output_dim]
    batch_size,
    input_dim,
    output_dim,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles one output element (one batch x one feature)
    program_id = tl.program_id(0)
    batch = program_id // output_dim
    feature = program_id % output_dim
    
    # Initialize accumulator as scalar
    acc = 0.0
    
    # Compute number of blocks needed for input dimension
    num_blocks = (input_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Loop over blocks of input dimension
    for k in range(num_blocks):
        # Compute offsets for this block
        k_offset = k * BLOCK_SIZE_K
        mask = k_offset + tl.arange(0, BLOCK_SIZE_K) < input_dim
        
        # Load a block of input data for this batch
        x_offset = batch * input_dim + k_offset
        x = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_SIZE_K), 
                   mask=mask, other=0.0)
        
        # Load a block of weights for this feature  
        w_offset = feature * input_dim + k_offset
        w = tl.load(weight_ptr + w_offset + tl.arange(0, BLOCK_SIZE_K), 
                   mask=mask, other=0.0)
        
        # Dot product for this block
        acc += tl.sum(x * w)
    
    # Add bias
    bias = tl.load(bias_ptr + feature)
    acc += bias
    
    # Store result
    out_offset = batch * output_dim + feature
    tl.store(out_ptr + out_offset, acc)

@torch.fx.wrap
def triton_linear(x, weight, bias):
    batch_size, input_dim = x.shape
    output_dim = weight.shape[0]
    
    out = torch.empty((batch_size, output_dim), dtype=torch.float32, device=x.device)
    
    # Block size for tiling along input dimension
    BLOCK_SIZE_K = 128
    
    # Calculate grid size - each CUDA block handles one output element
    grid_size = batch_size * output_dim
    
    linear_kernel[grid_size,](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        input_dim=input_dim,
        output_dim=output_dim,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return triton_linear