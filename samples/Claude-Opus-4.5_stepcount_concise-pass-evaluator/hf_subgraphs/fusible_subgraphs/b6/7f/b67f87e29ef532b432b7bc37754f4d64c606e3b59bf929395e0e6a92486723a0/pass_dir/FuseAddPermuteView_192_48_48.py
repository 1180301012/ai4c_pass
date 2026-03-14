import torch
import triton
import triton.language as tl

# Pattern matching function - matches add + permute + view for graph 2
# Input shape: [1, 2304, 192], output: [1, 192, 48, 48]
def pattern(in_0, in_1):
    """
    Match the pattern:
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.permute(0, 2, 1)
    tmp_2 = tmp_1.view(1, 192, 48, 48)
    """
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.permute(0, 2, 1)
    tmp_2 = tmp_1.view(1, 192, 48, 48)
    return tmp_2


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def add_permute_kernel_2(
    in0_ptr,
    in1_ptr,
    out_ptr,
    N,  # second dimension (spatial)
    C,  # third dimension (channels)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused add + permute kernel.
    Input: [1, N, C]
    Output: [1, C, N]
    
    For output[0, c, n], we read from input[0, n, c]
    Output linear index: c * N + n
    Input linear index: n * C + c
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert output linear index to (c, n)
    c = offsets // N
    n = offsets % N
    
    # Convert to input linear index
    in_idx = n * C + c
    
    # Load from both inputs and add
    x = tl.load(in0_ptr + in_idx, mask=mask, other=0.0)
    y = tl.load(in1_ptr + in_idx, mask=mask, other=0.0)
    result = x + y
    
    # Store to output (already in transposed layout)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def add_permute_view_fused_2(in_0, in_1):
    """
    Fused implementation of add + permute + view.
    Output shape: [1, 192, 48, 48]
    """
    # Input shape: [1, 2304, 192]
    batch = in_0.shape[0]
    N = in_0.shape[1]  # spatial dimension (H*W) = 2304
    C = in_0.shape[2]  # channel dimension = 192
    
    n_elements = batch * N * C
    
    # Output shape: [1, 192, 48, 48]
    out = torch.empty((batch, C, 48, 48), dtype=in_0.dtype, device=in_0.device)
    
    # Ensure inputs are contiguous
    in_0_contig = in_0.contiguous()
    in_1_contig = in_1.contiguous()
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_permute_kernel_2[grid](
        in_0_contig,
        in_1_contig,
        out,
        N,
        C,
        n_elements,
    )
    
    return out


# Replacement function - returns the fused kernel wrapper
def replacement_func():
    return add_permute_view_fused_2