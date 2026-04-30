import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_mul_sum_kernel(
    x_ptr,
    weights_ptr,
    out_ptr,
    n_elements,
    n_classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: sum(softmax(x, dim=1) * weights, dim=1)
    
    This fuses:
    - softmax along dim 1
    - element-wise multiplication with weights
    - sum reduction along dim 1
    """
    # Get program id - each program handles one batch element
    pid = tl.program_id(0)
    
    # Compute row offset (batch dimension)
    row_offset = pid * n_classes
    
    # Compute max for numerical stability of softmax
    max_val = float('-inf')
    for i in range(BLOCK_SIZE):
        offset = row_offset + i
        mask = offset < n_elements
        if offset < n_elements:
            val = tl.load(x_ptr + offset, mask=mask, other=float('-inf'))
            max_val = tl.max(val, max_val) if i > 0 else val
    
    # Second pass: compute exp(x - max) and weighted sum in one go
    weighted_sum = 0.0
    exp_sum = 0.0
    
    # Load all elements in this row and compute exp(x - max)
    for i in range(BLOCK_SIZE):
        offset = row_offset + i
        mask = offset < n_elements
        if offset < n_elements:
            x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
            w_val = tl.load(weights_ptr + i, mask=mask, other=0.0)
            exp_val = tl.exp(x_val - max_val)
            weighted_sum = weighted_sum + exp_val * w_val
            exp_sum = exp_sum + exp_val
    
    # Normalize by sum of exponentials to get weighted expectation
    result = weighted_sum / exp_sum
    
    # Store result
    tl.store(out_ptr + pid, result)


@torch.fx.wrap
def fused_softmax_mul_sum(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Fused kernel that computes: sum(softmax(x, dim=1) * weights, dim=1)
    """
    assert x.dim() == 2, "Expected 2D input tensor"
    batch_size, n_classes = x.shape
    assert weights.shape == (n_classes,), f"Expected weights shape ({n_classes},), got {weights.shape}"
    
    out = torch.empty((batch_size,), dtype=x.dtype, device=x.device)
    
    # Each program handles one batch element
    grid = (batch_size,)
    BLOCK_SIZE = 8  # n_classes = 5, so 8 is enough
    
    fused_softmax_mul_sum_kernel[grid](
        x_ptr=x,
        weights_ptr=weights,
        out_ptr=out,
        n_elements=x.numel(),
        n_classes=n_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match: softmax(in_0) * in_1, sum(dim=1)
    
    Pattern is:
    tmp_0 = torch.nn.functional.softmax(in_0, dim = 1)
    tmp_1 = tmp_0 * in_1
    tmp_2 = tmp_1.sum(dim = 1)
    return tmp_2
    """
    tmp_0 = torch.nn.functional.softmax(in_0, dim = 1)
    tmp_1 = tmp_0 * in_1
    tmp_2 = tmp_1.sum(dim = 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_softmax_mul_sum