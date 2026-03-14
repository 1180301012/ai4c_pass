import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match sigmoid pattern
    """
    tmp_1 = torch.sigmoid(in_0)
    return tmp_1


def replacement_args(in_0):
    """
    Extract arguments for the replacement function
    """
    return (in_0,)


@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast Sigmoid kernel optimized for small tensors
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Sigmoid: 1 / (1 + exp(-x))
    sigmoid_out = tl.sigmoid(x)
    
    # Store output
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)


@torch.fx.wrap
def triton_sigmoid(x):
    """
    Wrapper function to launch sigmoid kernel
    """
    x_contig = x.contiguous()
    n_elements = x_contig.numel()
    out = torch.empty_like(x_contig)
    
    # Use power-of-2 block size that fits the tensor
    # For small tensors, use smaller block sizes
    if n_elements <= 256:
        BLOCK_SIZE = 256
    elif n_elements <= 1024:
        BLOCK_SIZE = 1024
    elif n_elements <= 8192:
        BLOCK_SIZE = 8192
    else:
        BLOCK_SIZE = 8192
    
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    sigmoid_kernel[grid](
        x_ptr=x_contig,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    """
    Return the replacement function (not a call to it)
    """
    return triton_sigmoid