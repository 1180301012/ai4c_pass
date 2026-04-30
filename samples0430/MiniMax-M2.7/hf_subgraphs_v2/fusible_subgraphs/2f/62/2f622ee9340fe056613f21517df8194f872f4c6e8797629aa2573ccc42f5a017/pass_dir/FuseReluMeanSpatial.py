import torch
import triton
import triton.language as tl

# Autotune configuration for the ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    ReLU kernel: y = max(0, x)
    Each program processes a contiguous block of elements.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)


# Autotune configuration for the mean reduction kernel
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
    ],
    key=['N', 'C'],
)
@triton.jit  
def mean_reduce_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    """
    Reduction kernel for computing mean over spatial dimensions (H, W).
    Each program handles one channel of one batch element.
    """
    pid = tl.program_id(0)
    n_idx = pid // C
    c_idx = pid % C
    
    # Sum all H*W values for this channel
    sum_val = 0.0
    for h in range(H):
        for w in range(W):
            idx = n_idx * C * H * W + c_idx * H * W + h * W + w
            x = tl.load(input_ptr + idx)
            x = tl.where(x > 0, x, 0.0)  # ReLU computation
            sum_val = sum_val + x
    
    mean_val = sum_val / (H * W)
    
    # Store mean result [N, C, 1, 1]
    out_idx = n_idx * C * 1 * 1 + c_idx * 1 * 1
    tl.store(output_ptr + out_idx, mean_val)


@torch.fx.wrap
def fused_relu_mean(x: torch.Tensor) -> tuple:
    """
    Fused kernel that computes:
    - relu(x, inplace=True) -> returns the tensor with relu applied
    - mean(relu(x), dim=(2, 3), keepdim=True)
    
    Returns (relu_out, mean_out) to match the pattern's return structure.
    """
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    
    # Apply ReLU to get tmp_0
    relu_out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, relu_out, n_elements)
    
    # Compute mean over spatial dimensions [N, C, 1, 1]
    mean_out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    grid = (N * C,)
    mean_reduce_kernel[grid](relu_out, mean_out, N, C, H, W)
    
    return relu_out, mean_out


def pattern(in_0, in_1):
    """
    Match the pattern:
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_1 = in_0 // divisor  # scalar, varies: 8, 16, 32
    tmp_2 = torch.sym_sum([1, tmp_1])  # scalar operation
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_3)
    
    Note: in_0 operations are scalar and independent of the tensor computation.
    We optimize by fusing ReLU + mean.
    """
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_1 = in_0 // 8  # Matches the pattern structure
    tmp_2 = torch.sym_sum([1, tmp_1])
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_3)


def replacement_args(in_0, in_1):
    """
    Extract arguments for the fused kernel.
    in_0 is a scalar (sym_sum input), passed through for pattern matching completeness
    in_1 is the main tensor to optimize
    """
    return (in_0, in_1)


def replacement_func():
    """
    Returns a function that performs the fused ReLU + mean operation.
    """
    return fused_relu_mean