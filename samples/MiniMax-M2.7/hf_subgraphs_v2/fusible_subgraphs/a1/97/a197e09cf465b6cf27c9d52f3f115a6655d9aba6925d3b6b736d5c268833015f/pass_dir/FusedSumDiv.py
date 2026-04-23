import torch
import triton
import triton.language as tl

@triton.jit
def optimized_sum_kernel(
    x_ptr,
    out_ptr,
    n_elements_per_sum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized reduction sum kernel using sequential reduction.
    Each program computes sum of BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)
    
    # Calculate base offset
    base_offset = pid * n_elements_per_sum
    
    # Sequential reduction
    sum_val = 0.0
    for i in range(BLOCK_SIZE):
        x = tl.load(x_ptr + base_offset + i)
        sum_val += x
    
    # Store result
    tl.store(out_ptr + pid, sum_val)


@torch.fx.wrap
def optimized_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized sum over last dimension.
    Input: [1, 16, 196, 196]
    Output: [1, 16, 196]
    """
    B, H, W1, W2 = x.shape
    
    out = torch.empty(B, H, W1, dtype=x.dtype, device=x.device)
    
    # Grid: B * H * W1 programs (3136 programs for 1x16x196)
    n_elements = B * H * W1
    
    # Use W2 threads per program (196 elements)
    BLOCK_SIZE = W2
    
    optimized_sum_kernel[(n_elements,)](
        x_ptr=x,
        out_ptr=out,
        n_elements_per_sum=W2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0):
    """
    Pattern: just sum
    """
    tmp_0 = in_0.sum(dim=-1)
    return tmp_0


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return optimized_sum