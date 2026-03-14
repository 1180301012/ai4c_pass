import torch
import triton
import triton.language as tl


# Autotune configurations for different input sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 4096}, num_stages=4, num_warps=16),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    stride_b: tl.constexpr,
    stride_s: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel that computes: softmax(x - max(x, dim=-1, keepdim=True))
    
    This fuses 4 operations:
    1. torch.max(in_0, -1, keepdim=True)
    2. expand_as (broadcast)
    3. subtraction (max - x)
    4. softmax
    """
    # Get position
    pid = tl.program_id(0)
    
    # Calculate which batch this program handles
    batch_idx = pid
    if batch_idx >= batch_size:
        return
    
    # Base pointer for this batch
    base = batch_idx * stride_b
    
    # First pass: compute max
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < seq_len
    
    # Load all values for this batch row
    values = tl.load(input_ptr + base + offs_n * 1, mask=mask, other=-float('inf'))
    
    # Compute max using Triton reduction
    max_val = tl.max(values, axis=0)
    
    # Second pass: compute exp(x - max) and sum
    # Compute shifted values: x - max
    shifted = values - max_val
    
    # Compute exp(shifted)
    exp_shifted = tl.exp(shifted)
    
    # Sum of exp values
    sum_exp = tl.sum(exp_shifted, axis=0)
    
    # Compute softmax: exp(x - max) / sum(exp(x - max))
    softmax_vals = exp_shifted / sum_exp
    
    # Store result
    tl.store(output_ptr + base + offs_n * 1, softmax_vals, mask=mask)


@torch.fx.wrap
def fused_softmax(x):
    """
    Fused numerically stable softmax kernel.
    Computes: softmax(x - max(x, dim=-1, keepdim=True))
    """
    batch_size, seq_len = x.shape[0], x.shape[1]
    
    # Output tensor
    out = torch.empty_like(x)
    
    # Grid: one program per batch
    grid = (batch_size,)
    
    # Launch kernel
    fused_softmax_kernel[grid](
        x,
        out,
        batch_size,
        seq_len,
        x.stride(0),
        x.stride(1),
        BLOCK_M=1,
        BLOCK_N=seq_len,
    )
    
    return out


def pattern(in_0, in_1):
    """ 
    Match the numerically stable softmax pattern:
    max(in_0, dim=-1, keepdim=True) -> expand_as(in_0) -> subtract -> softmax
    
    Also preserve the view operation on in_1 since it's in the return values.
    """
    # Softmax computation on in_0
    # Step 1: max along last dimension with keepdim
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    
    # Step 2: extract values from the tuple (max returns (values, indices))
    tmp_1 = tmp_0[0]
    
    # Step 3: expand max to match in_0 shape (broadcast)
    tmp_2 = tmp_1.expand_as(in_0)
    
    # Step 4: subtract original from expanded max (numerically stable shift)
    tmp_3 = tmp_2 - in_0
    
    # Step 5: apply softmax along last dimension
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    
    # View computation on in_1 - preserve this as it appears in the return
    # The view reshapes from [B, C, H, W] to [B, C, H*W]
    tmp_5 = in_1.view(in_1.shape[0], 512, -1)
    
    return tmp_4, tmp_5


def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement"""
    return (in_0, in_1)


def replacement_func():
    """Return the fused kernel function that handles both softmax and view"""
    def fused_kernel(in_0, in_1):
        # Compute fused softmax on in_0
        softmax_out = fused_softmax(in_0)
        # View in_1 to match the original shape (batch, 512, -1)
        view_out = in_1.view(in_1.shape[0], 512, -1)
        return softmax_out, view_out
    return fused_kernel