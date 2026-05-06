import torch
import triton
import triton.language as tl

def pattern(input, weight, bias):
    return torch.nn.functional.layer_norm(input, (384,), weight, bias, 1e-12)

def replacement_args(input, weight, bias):
    return (input, weight, bias)

@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    eps,
    output_ptr,
    N,
    S,
    F,
    BLOCK_SIZE: tl.constexpr,
):
    # Token index and block index
    pid = tl.program_id(0)
    s = pid  # Sequence index
    
    # Feature block index
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load input features for this token and block
    input_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        input_block[i] = tl.load(input_ptr + (s * F + i), mask=offsets < BLOCK_SIZE)
    
    # Compute mean and variance for this block (simplified for this example)
    mean = tl.mean(input_block)
    variance = tl.mean((input_block - mean)**2)
    
    # Normalize inputs
    norm = (input_block - mean) / (tl.sqrt(variance) + eps)
    
    # Apply weight and bias
    output_block = norm * tl.load(weight_ptr) + tl.load(bias_ptr)
    
    # Store output
    for i in range(BLOCK_SIZE):
        tl.store(output_ptr + (s * F + i), output_block[i], mask=offsets < BLOCK_SIZE)

@torch.fx.wrap
def layer_norm_wrapper(input, weight, bias):
    N, S, F = input.shape
    BLOCK_SIZE = 128  # Optimized block size for 384 features
    num_programs = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input)
    
    layer_norm_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        eps=1e-12,
        output_ptr=output,
        N=N,
        S=S,
        F=F,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return layer_norm_wrapper