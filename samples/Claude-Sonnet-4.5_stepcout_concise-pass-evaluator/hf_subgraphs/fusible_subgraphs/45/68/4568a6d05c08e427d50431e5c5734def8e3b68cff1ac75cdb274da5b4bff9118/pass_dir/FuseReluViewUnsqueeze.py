import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Match the pattern: ReLU -> view -> unsqueeze
    Returns both the final reshaped output and the ReLU output
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    batch_size = in_0.shape[0]
    tmp_1 = tmp_0.view(batch_size, 512, 4096)
    tmp_2 = tmp_1.unsqueeze(1)
    return (tmp_2, tmp_0)

def replacement_args(in_0):
    return (in_0,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized in-place ReLU kernel with autotuning"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(x, 0)
    output = tl.maximum(x, 0.0)
    
    # Store result (in-place)
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_relu_reshape(in_0):
    """
    Optimized implementation that:
    1. Applies ReLU in-place using Triton kernel
    2. Reshapes using native PyTorch operations (metadata-only)
    """
    n_elements = in_0.numel()
    
    # Grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Apply in-place ReLU
    relu_kernel[grid](
        in_0,
        in_0,  # in-place operation
        n_elements,
    )
    
    # Apply view and unsqueeze (metadata operations)
    batch_size = in_0.shape[0]
    tmp_1 = in_0.view(batch_size, 512, 4096)
    tmp_2 = tmp_1.unsqueeze(1)
    
    return (tmp_2, in_0)

def replacement_func():
    return optimized_relu_reshape