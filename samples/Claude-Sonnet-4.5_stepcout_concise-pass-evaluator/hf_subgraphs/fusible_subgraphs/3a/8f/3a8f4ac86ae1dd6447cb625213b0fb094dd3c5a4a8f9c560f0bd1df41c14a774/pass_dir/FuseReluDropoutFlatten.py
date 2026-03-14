import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Pattern to match: ReLU -> Dropout (p=0.0) -> Flatten
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_relu_dropout_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that applies ReLU and handles reshape.
    Dropout with p=0.0 is a no-op, so it's omitted.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    out = tl.maximum(x, 0.0)
    
    # Store output (flatten is handled by output shape)
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_dropout_flatten(x):
    """
    Wrapper function that sets up and launches the fused kernel.
    """
    # Calculate output shape after flatten(1, -1)
    batch_size = x.shape[0]
    flattened_size = x.numel() // batch_size
    output_shape = (batch_size, flattened_size)
    
    # Allocate output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_relu_dropout_flatten_kernel[grid](
        x,
        out,
        n_elements,
    )
    
    return out

def replacement_func():
    return fused_relu_dropout_flatten