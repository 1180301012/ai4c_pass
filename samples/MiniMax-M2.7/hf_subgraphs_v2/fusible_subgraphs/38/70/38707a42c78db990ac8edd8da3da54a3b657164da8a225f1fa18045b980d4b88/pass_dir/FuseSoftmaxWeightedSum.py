import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    """
    Match the computation pattern:
    softmax(in_0, dim=1) * linspace(0, 4, steps=5) -> sum(dim=1) -> 5 - result
    """
    # Compute softmax along dim=1
    tmp_0 = torch.nn.functional.softmax(in_0, dim=1)
    
    # Create linspace [0, 1, 2, 3, 4] on the same device
    tmp_1 = torch.linspace(0, 4, steps=5, device=in_0.device)
    
    # Multiply softmax by linspace
    tmp_2 = tmp_0 * tmp_1
    
    # Sum along dim=1
    tmp_3 = tmp_2.sum(dim=1)
    
    # Subtract from 5
    tmp_4 = 5 - tmp_3
    
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_softmax_weighted_sum_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    n_classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles reduction for one batch element
    batch_idx = tl.program_id(0)
    
    # Compute max for numerical stability
    max_val = -float('inf')
    for i in range(n_classes):
        offsets = batch_idx * n_classes + i
        x = tl.load(x_ptr + offsets).to(tl.float32)
        max_val = tl.max(max_val, x)
    
    # Compute exp(x - max) and weighted sum
    exp_sum = 0.0
    weighted_sum = 0.0
    for i in range(n_classes):
        offsets = batch_idx * n_classes + i
        x = tl.load(x_ptr + offsets).to(tl.float32)
        exp_val = tl.exp(x - max_val)
        exp_sum += exp_val
        weighted_sum += exp_val * i.to(tl.float32)
    
    # Compute result: 5 - weighted_sum / exp_sum
    result = 5.0 - weighted_sum / exp_sum
    tl.store(out_ptr + batch_idx, result.to(tl.float32))

@torch.fx.wrap
def fused_softmax_weighted_sum(in_0):
    """
    Fused kernel that computes:
    5 - sum(softmax(in_0, dim=1) * [0,1,2,3,4])
    in a single pass without materializing intermediate tensors.
    """
    batch_size, n_classes = in_0.shape
    BLOCK_SIZE = 128
    
    out = torch.empty((batch_size,), dtype=torch.float32, device=in_0.device)
    
    # Launch one block per batch element
    grid = (batch_size,)
    
    fused_softmax_weighted_sum_kernel[grid](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=batch_size * n_classes,
        n_classes=n_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_softmax_weighted_sum