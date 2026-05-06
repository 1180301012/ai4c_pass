import torch
import triton
import triton.language as tl

def pattern(in_0):
    c1 = torch.tensor(256.0, device=in_0.device, dtype=in_0.dtype)
    c2 = torch.tensor(0.5)
    c3 = c1 ** c2
    in_0_scaled = in_0 / c3
    c4 = torch.tensor(0.05, device=in_0.device, dtype=in_0.dtype)
    in_0_scaled = in_0_scaled / c4
    return in_0_scaled.softmax(dim=-1)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    scaled = input * 1.25
    
    # Compute softmax via standard Triton implementation
    # (This would be fully optimized in production, but for this case we use simplified version)
    max_val = tl.maximum(scaled, axis=-1)
    exp_vals = tl.exp(scaled - max_val)
    sum_exp = tl.sum(exp_vals, axis=-1)
    softmax_vals = exp_vals / sum_exp
    
    tl.store(output_ptr + offsets, softmax_vals, mask=mask)

def kernel_wrapper(input):
    output = torch.empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    
    optimized_kernel[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,](
        input_ptr=input,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def replacement_func():
    return kernel_wrapper