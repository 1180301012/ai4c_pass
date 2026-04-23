import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.ops.aten.relu.default(in_0)
    tmp_1 = torch.ops.aten.dropout.default(tmp_0, 0.1, False)
    return (tmp_1, tmp_0)

def replacement_args(in_0):
    return (in_0,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_relu_dropout2d_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    # relu: max(x, 0)
    # dropout2d with training=False is identity, so no additional computation needed
    result = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_relu_dropout2d(in_0):
    N = in_0.numel()
    grid = ((N + 4096 - 1) // 4096,)  # upper bound for grid size; autotune will handle BLOCK_SIZE
    
    output = torch.empty_like(in_0)
    
    fused_relu_dropout2d_kernel[grid](
        input_ptr=in_0,
        output_ptr=output,
        n_elements=N,
    )
    
    # Both outputs are the same tensor since dropout2d with training=False is identity
    return (output, output)

def replacement_func():
    return fused_relu_dropout2d