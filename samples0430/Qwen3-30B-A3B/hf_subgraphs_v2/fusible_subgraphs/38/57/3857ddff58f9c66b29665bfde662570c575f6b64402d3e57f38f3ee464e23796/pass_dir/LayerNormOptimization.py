import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches layer_norm with any normalized_shape, weight, bias, and eps
# The pattern must exactly match the model.py operations
# Note: layer_norm in model.py uses tuple (768,) or (16,) for normalized_shape
# We use normalized_shape as a variable to capture the shape

def pattern(x, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

# Argument extraction function
# We extract all necessary arguments for the replacement
# The D (feature dimension) is normalized_shape[0]
def replacement_args(x, normalized_shape, weight, bias, eps):
    D = normalized_shape[0]  # D is the feature dimension from the tuple
    return (x, weight, bias, eps, D)


# Triton kernel for efficient layer normalization
@triton.jit
@triton.autotune(
    configs=[
        triton.autotune.Config({'BLOCK_SIZE_D': 128}, num_stages=3, num_warps=4),
        triton.autotune.Config({'BLOCK_SIZE_D': 64}, num_stages=3, num_warps=4),
        triton.autotune.Config({'BLOCK_SIZE_D': 32}, num_stages=3, num_warps=4),
    ],
    key=['D']
)
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, D, eps,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Row index for the current block
    row_idx = tl.program_id(0)
    row_start = row_idx * D

    # Block index for the current segment of the row
    block_idx = tl.program_id(1)
    element_start = block_idx * BLOCK_SIZE_D

    # Mask for valid elements in the current segment
    mask = tl.arange(0, BLOCK_SIZE_D) + element_start < D

    # Load a block of input values
    x_block = tl.load(x_ptr + row_start + element_start, mask=mask, other=0.0)

    # Compute mean and variance for the entire row (only by first block in the row)
    if block_idx == 0:
        sum_val = 0.0
        sum_sq = 0.0
        for i in range(D):
            x_i = tl.load(x_ptr + row_start + i)
            sum_val += x_i
            sum_sq += x_i * x_i
        
        mean = sum_val / D
        var = sum_sq / D - mean * mean
        # Broadcast mean and variance to all threads in the row
        tl.store(tl.shared_memory(8), (mean, var), 2)

    # Synchronize to ensure mean/var are written before use
    tl.sync_threads()

    # Load mean and variance from shared memory
    mean, var = tl.load(tl.shared_memory(8), 2)

    # Normalize and apply scale/bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x_block - mean) * inv_std
    weight_block = tl.load(weight_ptr + element_start, mask=mask, other=0.0)
    bias_block = tl.load(bias_ptr + element_start, mask=mask, other=0.0)
    out_block = x_norm * weight_block + bias_block

    # Store normalized output
    tl.store(out_ptr + row_start + element_start, out_block, mask=mask)


# Triton kernel wrapper
@torch.fx.wrap
@triton.autotune(
    configs=[
        triton.autotune.Config({'BLOCK_SIZE_D': 128}, num_stages=3, num_warps=4),
        triton.autotune.Config({'BLOCK_SIZE_D': 64}, num_stages=3, num_warps=4),
        triton.autotune.Config({'BLOCK_SIZE_D': 32}, num_stages=3, num_warps=4),
    ],
    key=['D']
)
def layer_norm_wrapper(x, weight, bias, eps, D):
    N = x.shape[0]  # Number of rows
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Grid dimensions: (N, ceil(D / BLOCK_SIZE_D))
    grid_n = N
    grid_d = (D + 128 - 1) // 128
    
    # Launch the kernel
    layer_norm_kernel[(grid_n, grid_d)](
        x_ptr=x, weight_ptr=weight, bias_ptr=bias, out_ptr=out,
        N=N, D=D, eps=eps,
        BLOCK_SIZE_D=128,
    )
    
    return out


# Replacement function
# Must return the kernel wrapper function (not called)
def replacement_func():
    return layer_norm_wrapper