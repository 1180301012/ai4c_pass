import torch
import triton
import triton.language as tl

def pattern(in_0):
    return in_0.flatten(1, -1)

def replacement_args(in_0):
    return (in_0, "optimize_flatten")

# Autotuning configurations for the kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=1, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def relu_flatten_fused(x, route=""):
    """
    Optimized ReLU + Flatten using contiguous memory access.
    """
    orig_shape = x.shape
    batch_size = orig_shape[0]
    num_features = orig_shape[1] * orig_shape[2] * orig_shape[3]
    n_elements = batch_size * num_features
    
    # Make input contiguous for better memory access
    x_contig = x.contiguous()
    
    # Allocate output
    out = torch.empty(batch_size, num_features, dtype=x.dtype, device=x.device)
    
    # Use grid size based on total elements
    grid = lambda meta: (triton.next_power_of_2(n_elements),)
    
    relu_kernel[grid](
        x_ptr=x_contig,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    return out

def replacement_func():
    return relu_flatten_fused