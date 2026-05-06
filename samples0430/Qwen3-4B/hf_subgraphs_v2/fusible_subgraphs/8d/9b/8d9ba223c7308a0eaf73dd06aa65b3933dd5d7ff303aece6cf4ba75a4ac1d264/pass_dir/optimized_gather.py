import torch
import triton
import triton.language as tl


def pattern(linear, indices):
    \n    # Must match EXACTLY the pattern from model.py\n    # linear: output of linear layer (consecutive indices)
    # indices: shape [K]
    temp = linear.view(-1, 12)
    out = temp[indices.view(-1)]
    return out


def replacement_args(linear, indices):
    return (linear, indices)


@triton.jit
@triton.heuristics(
    enable_tensor_parallelism=False,
    full_kernel=True,
)

def optimized_gather_kernel(
    linear_ptr,  # [N, 12] float32
    indices_ptr,  # [K] int32
    out_ptr,      # [K, 12] float32
    n_indices,    # K
    n_rows,       # N
    n_cols,       # 12
    BLOCK_SIZE: tl.constexpr,
):
    
    # Compute block index
    offset = tl.program_id(0) * BLOCK_SIZE
    # Only process within bounds
    if offset >= n_indices:
        return
    
    # Compute local thread index
    i = tl.arange(0, BLOCK_SIZE)[None, :]  # [BLOCK_SIZE, 1]
    # Fill with actual index values (first DIM)
    idx = tl.load(indices_ptr + offset, tl.int32)
    
    # Compute row offset in linear (contiguous)
    row_offset = idx * n_cols
    
    # Load row (with memory coalescing)
    row = tl.zeros((BLOCK_SIZE, n_cols), dtype=tl.float32)
    for j in range(BLOCK_SIZE):
        row[j] = tl.load(linear_ptr + row_offset + j)
    
    # Store to output
    tl.store(out_ptr + offset, row)


@torch.fx.wrap

def kernel_wrapper(linear, indices):
    n_indices = indices.numel()
    n_rows = linear.shape[0]
    n_cols = 12
    
    out = torch.empty((n_indices, n_cols), dtype=torch.float32)
    
    # Launch kernel with appropriate grid
    grid = ( (n_indices + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    optimized_gather_kernel[grid](
        linear_ptr=linear,
        indices_ptr=indices,
        out_ptr=out,
        n_indices=n_indices,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=128,
    )
    
    return out


def replacement_func():
    return kernel_wrapper