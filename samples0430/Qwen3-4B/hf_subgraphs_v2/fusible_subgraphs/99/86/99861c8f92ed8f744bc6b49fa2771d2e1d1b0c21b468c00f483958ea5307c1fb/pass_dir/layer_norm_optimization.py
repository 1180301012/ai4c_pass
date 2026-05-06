import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return tmp_2, tmp_4
def replacement_args(in_0, in_1, in_2, in_3):
    return in_0, in_1, in_2, in_3

@triton.jit
def optimized_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the index in the feature dimension (last dimension)
    # We process 1024 features with BLOCK_SIZE alignment
    idx = tl.program_id(0)
    offs = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load input data (batch, seq, features)
    input = tl.load(input_ptr + offs, mask=mask, other=0.0)
    
    # Calculate mean and std across features
    mean = tl.reduce(input, tl.arange(0, BLOCK_SIZE), 'sum') / BLOCK_SIZE
    var = tl.reduce((input - mean) ** 2, tl.arange(0, BLOCK_SIZE), 'sum')
    std = tl.sqrt(var / BLOCK_SIZE + eps)
    
    # Normalize and apply weight/bias
    normalized = (input - mean) / (std + eps)
    scaled = normalized * tl.load(weight_ptr + offs) + tl.load(bias_ptr + offs)
    
    # Store result
    tl.store(output_ptr + offs, scaled, mask=mask)

@torch.fx.wrap
def optimized_layer_norm_wrapper(input, weight, bias):
    n_elements = input.numel()
    BLOCK_SIZE = 256  # Optimized tile size
    output = torch.empty_like(input)
    
    optimized_layer_norm_kernel[
        (1, ),  # 1 program per block
    ](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def replacement_func():
    return optimized_layer_norm_wrapper