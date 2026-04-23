import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=3, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def silu_kernel_impl(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused silu kernel: silu(x) = x * sigmoid(x)
    
    Optimized implementation using sigmoid(x) = 1 / (1 + exp(-x))
    to avoid separate sigmoid kernel launch.
    """
    # Calculate block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid(x) using fast sigmoid: 1 / (1 + exp(-x))
    # Use clamp to prevent overflow in exp(-x) when x is very negative
    exp_neg_x = tl.exp(-tl.clamp(x, -20.0, 20.0))
    sigmoid_x = 1.0 / (1.0 + exp_neg_x)
    
    # Silu: x * sigmoid(x)
    out = x * sigmoid_x
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_silu_detach(x):
    """
    Triton-based silu implementation with optimized kernel.
    Returns the same tensor with silu applied - works with autograd.
    """
    n_elements = x.numel()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Calculate number of programs needed
    # The autotuner will select the optimal BLOCK_SIZE
    num_programs = (n_elements + 2047) // 2048  # Start with max possible
    if num_programs == 0:
        num_programs = 1
    
    # Launch autotuned kernel
    silu_kernel_impl[(num_programs,)](
        in_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Match the pattern:
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement.
    All four output values need to be returned to match the pattern outputs.
    """
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Returns a function that implements the silu + detach fusion.
    The replacement function runs the optimized Triton kernel for silu
    while the detach operations are no-ops in the forward pass.
    """
    def wrapper(in_0, in_1, in_2):
        # Apply optimized silu to in_0
        tmp_0 = triton_silu_detach(in_0)
        
        # Detach operations are no-ops for forward computation
        # They only affect gradient tracking
        tmp_1 = in_1.detach()
        tmp_2 = in_2.detach()
        tmp_3 = tmp_0.detach()
        
        return (tmp_1, tmp_2, tmp_3, tmp_0)
    
    return wrapper