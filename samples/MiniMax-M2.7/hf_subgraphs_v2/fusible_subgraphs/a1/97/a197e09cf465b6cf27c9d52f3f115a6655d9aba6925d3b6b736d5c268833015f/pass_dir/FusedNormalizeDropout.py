import torch
import triton
import triton.language as tl

@triton.jit
def fused_normalize_dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    - sum over last dimension (W)
    - normalize by dividing by sum
    - dropout with p=0 (no-op)
    
    Input shape: [B, H, W]
    Output shape: [B, H, W]
    """
    # Each program handles a batch-head combination
    pid = tl.program_id(0)
    num_bh = n_elements
    
    # Calculate which batch-head this program handles
    bh_idx = pid
    b = bh_idx // H
    h = bh_idx % H
    
    # Compute sum over W dimension for this batch-head
    sum_val = 0.0
    for w in range(W):
        x_idx = b * H * W + h * W + w
        x = tl.load(x_ptr + x_idx)
        sum_val += x
    
    # Normalize epsilon to avoid division by zero
    eps = 1e-8
    norm_factor = sum_val + eps
    
    # Store results
    for w in range(W):
        x_idx = b * H * W + h * W + w
        out_idx = b * H * W + h * W + w
        x = tl.load(x_ptr + x_idx)
        out = x / norm_factor
        tl.store(out_ptr + out_idx, out)


@torch.fx.wrap
def fused_normalize_dropout(x: torch.Tensor) -> torch.Tensor:
    """
    Fused implementation of:
    1. x.sum(dim=-1) - sum over last dimension
    2. unsqueeze(-1) - add dimension for broadcasting
    3. x / sum - normalize
    4. dropout(x, p=0.0) - no-op since p=0
    
    Input: [B, H, W] - typically [1, 16, 196]
    Output: [B, H, W]
    """
    B, H, W = x.shape
    
    out = torch.empty_like(x)
    
    n_elements = B * H
    grid = (n_elements,)
    
    fused_normalize_dropout_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        H=H,
        W=W,
        BLOCK_SIZE=1,  # We're processing W elements per program
    )
    
    return out


def pattern(in_0):
    """
    Match the pattern from convit_base model.py:
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = in_0 / tmp_1  (original uses in_0 /= tmp_1, but FX traces this as /)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return tmp_3
    """
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = in_0 / tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_normalize_dropout