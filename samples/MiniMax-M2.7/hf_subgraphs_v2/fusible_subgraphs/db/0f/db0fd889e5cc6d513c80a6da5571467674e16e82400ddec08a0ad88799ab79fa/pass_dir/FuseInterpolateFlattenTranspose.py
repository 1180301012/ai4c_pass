import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Match a simple transpose operation
    """
    out = x.transpose(1, 2)
    return out


def replacement_args(x):
    return (x,)


@triton.jit
def triton_transpose_kernel(
    input_ptr,
    output_ptr,
    dim0_size: tl.constexpr,
    dim1_size: tl.constexpr,
    dim2_size: tl.constexpr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for transpose(1, 2) on 3D tensors [B, M, N] -> [B, N, M]"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert linear index to 3D (batch, m, n)
    # total_elements = B * M * N
    # linear_idx = b * M * N + m * N + n
    bMN = dim1_size * dim2_size
    batch = offsets // bMN
    remainder = offsets % bMN
    m = remainder // dim2_size
    n = remainder % dim2_size
    
    # Load from input (contiguous row-major [B, M, N])
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute output offset: b * N * M + n * M + m
    nM = dim2_size * dim1_size  # N * M = total_elements / B
    out_offset = batch * nM + n * dim1_size + m
    out_mask = offsets < (total_elements // (dim1_size * dim2_size)) * (dim2_size * dim1_size)
    
    # Store to output (contiguous row-major [B, N, M])
    tl.store(output_ptr + out_offset, x, mask=out_mask)


@torch.fx.wrap
def triton_transpose(x):
    """Triton-based transpose(1, 2) optimization"""
    # x has shape [B, M, N], output has shape [B, N, M]
    assert x.dim() == 3, "Only 3D tensors supported"
    B, M, N = x.shape
    
    n_elements = B * M * N
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with transposed shape using allowed operation
    out = torch.empty((B, N, M), dtype=x.dtype, device=x.device)
    
    # Launch the Triton kernel
    triton_transpose_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        dim0_size=B,
        dim1_size=M,
        dim2_size=N,
        total_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return triton_transpose