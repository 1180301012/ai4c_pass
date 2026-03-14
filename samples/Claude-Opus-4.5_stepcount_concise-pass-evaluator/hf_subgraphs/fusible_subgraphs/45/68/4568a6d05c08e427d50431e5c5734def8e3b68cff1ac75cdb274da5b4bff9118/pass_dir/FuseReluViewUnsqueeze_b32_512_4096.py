import torch
import triton
import triton.language as tl

# Pattern to match: ReLU -> view -> unsqueeze for batch size 32
def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.view(32, 512, 4096)
    tmp_2 = tmp_1.unsqueeze(1)
    return (tmp_2, tmp_0)

def replacement_args(in_0):
    return (in_0,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel_b32(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_view_unsqueeze_b32(in_0):
    N = in_0.numel()
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Launch ReLU kernel
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    relu_kernel_b32[grid](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=N,
    )
    
    # Apply view and unsqueeze (these are just reshape operations, no data copy)
    tmp_0 = out
    tmp_1 = tmp_0.view(32, 512, 4096)
    tmp_2 = tmp_1.unsqueeze(1)
    
    return (tmp_2, tmp_0)

def replacement_func():
    return fused_relu_view_unsqueeze_b32