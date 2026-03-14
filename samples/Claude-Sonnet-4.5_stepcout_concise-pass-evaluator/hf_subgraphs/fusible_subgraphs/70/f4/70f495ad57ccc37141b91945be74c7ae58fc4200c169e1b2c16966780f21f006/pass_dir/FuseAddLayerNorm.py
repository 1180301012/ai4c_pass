import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern to match: Add + LayerNorm
    """
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1280,), in_1, in_0, 1e-06)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['hidden_dim'],
)
@triton.jit
def fused_add_ln_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, seq_len, hidden_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for Add + LayerNorm
    Each program processes one row (one sequence position)
    """
    row_idx = tl.program_id(0)
    
    # Base offset for this row
    row_start = row_idx * hidden_dim
    
    # Compute mean
    mean = 0.0
    for off in range(0, hidden_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_dim
        
        a = tl.load(in_2_ptr + row_start + cols, mask=mask, other=0.0)
        b = tl.load(in_3_ptr + row_start + cols, mask=mask, other=0.0)
        val = a + b
        mean += tl.sum(tl.where(mask, val, 0.0))
    
    mean = mean / hidden_dim
    
    # Compute variance
    var = 0.0
    for off in range(0, hidden_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_dim
        
        a = tl.load(in_2_ptr + row_start + cols, mask=mask, other=0.0)
        b = tl.load(in_3_ptr + row_start + cols, mask=mask, other=0.0)
        val = a + b
        diff = tl.where(mask, val - mean, 0.0)
        var += tl.sum(diff * diff)
    
    var = var / hidden_dim
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and write output
    for off in range(0, hidden_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_dim
        
        a = tl.load(in_2_ptr + row_start + cols, mask=mask, other=0.0)
        b = tl.load(in_3_ptr + row_start + cols, mask=mask, other=0.0)
        val = a + b
        
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        
        out = (val - mean) * rstd * w + bias
        tl.store(out_ptr + row_start + cols, out, mask=mask)


@torch.fx.wrap
def fused_add_ln(in_0, in_1, in_2, in_3):
    """
    Wrapper function for the fused kernel
    """
    bias = in_0
    weight = in_1
    
    batch, seq_len, hidden_dim = in_2.shape
    
    # Output shape: same as input
    output = torch.empty_like(in_2)
    
    # Grid: one program per row (total_rows = batch * seq_len)
    n_rows = batch * seq_len
    grid = (n_rows,)
    
    # Launch kernel with autotune
    eps = 1e-06
    
    fused_add_ln_kernel[grid](
        in_2, in_3, weight, bias, output,
        batch, seq_len, hidden_dim,
        eps=eps,
    )
    
    return output


def replacement_func():
    return fused_add_ln