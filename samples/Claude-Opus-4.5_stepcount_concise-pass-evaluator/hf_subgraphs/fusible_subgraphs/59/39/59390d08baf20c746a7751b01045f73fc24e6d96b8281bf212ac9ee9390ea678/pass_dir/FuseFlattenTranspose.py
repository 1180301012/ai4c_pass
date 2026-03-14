import torch
import triton
import triton.language as tl

# Pattern to match: flatten(2) followed by transpose(1, 2)
def pattern(x):
    flat = x.flatten(2)
    transposed = flat.transpose(1, 2)
    return transposed

def replacement_args(x):
    return (x,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
    ],
    key=['channels', 'spatial'],
)
@triton.jit
def transpose_2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    spatial,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (num_blocks_m, num_blocks_n, batch_size)
    pid_batch = tl.program_id(2)
    pid_m = tl.program_id(0)  # along channels dimension
    pid_n = tl.program_id(1)  # along spatial dimension
    
    # Calculate offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for bounds checking
    mask_m = offs_m < channels
    mask_n = offs_n < spatial
    
    # Input: [batch, channels, spatial] - stored row-major
    input_ptrs = input_ptr + pid_batch * channels * spatial + offs_m[:, None] * spatial + offs_n[None, :]
    
    # Output: [batch, spatial, channels] - stored row-major
    output_ptrs = output_ptr + pid_batch * spatial * channels + offs_n[:, None] * channels + offs_m[None, :]
    
    # Load tile [BLOCK_M, BLOCK_N] from input
    load_mask = mask_m[:, None] & mask_n[None, :]
    tile = tl.load(input_ptrs, mask=load_mask, other=0.0)
    
    # Store transposed tile [BLOCK_N, BLOCK_M] to output
    store_mask = mask_n[:, None] & mask_m[None, :]
    tl.store(output_ptrs, tl.trans(tile), mask=store_mask)


@torch.fx.wrap
def flatten_transpose_kernel(x):
    # Get dimensions
    batch_size = x.shape[0]
    channels = x.shape[1]
    spatial = 1
    for i in range(2, len(x.shape)):
        spatial *= x.shape[i]
    
    # Ensure input is contiguous and reshape to 3D
    x_contig = x.contiguous()
    x_flat = x_contig.view(batch_size, channels, spatial)
    
    # Allocate output tensor [batch, spatial, channels]
    output = torch.empty(batch_size, spatial, channels, device=x.device, dtype=x.dtype)
    
    # Launch kernel with autotune
    grid = lambda meta: (
        triton.cdiv(channels, meta['BLOCK_M']),
        triton.cdiv(spatial, meta['BLOCK_N']),
        batch_size
    )
    
    transpose_2d_kernel[grid](
        x_flat,
        output,
        batch_size,
        channels,
        spatial,
    )
    
    return output


def replacement_func():
    return flatten_transpose_kernel