import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern: layer_norm + transpose
    in_0: bias [768]
    in_1: weight [768]
    in_2: input [B, 196, 768]
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.layer_norm(in_2, (768,), tmp_1, tmp_0, 1e-05)
    tmp_3 = tmp_2.transpose(-1, -2)
    return (tmp_3,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['seq_len', 'hidden_dim'],
)
@triton.jit
def layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (sequence position)
    row_idx = tl.program_id(0)
    
    if row_idx >= batch_size * seq_len:
        return
    
    # Input offset
    row_start = row_idx * hidden_dim
    
    # Compute mean
    mean = 0.0
    for off in range(0, hidden_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_dim
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
        mean += tl.sum(tl.where(mask, x, 0.0))
    mean = mean / hidden_dim
    
    # Compute variance
    var = 0.0
    for off in range(0, hidden_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_dim
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff)
    var = var / hidden_dim
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and write
    for off in range(0, hidden_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_dim
        
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        
        out = (x - mean) * rstd * weight + bias
        tl.store(output_ptr + row_start + cols, out, mask=mask)

@torch.fx.wrap
def fused_layernorm_transpose(in_0, in_1, in_2):
    """
    Optimized layer normalization + transpose
    in_0: bias [hidden_dim]
    in_1: weight [hidden_dim]
    in_2: input [batch_size, seq_len, hidden_dim]
    """
    batch_size, seq_len, hidden_dim = in_2.shape
    
    # Step 1: Apply layer norm with Triton (writes in non-transposed layout)
    normalized = torch.empty_like(in_2)
    eps = 1e-05
    
    grid = (batch_size * seq_len,)
    layernorm_kernel[grid](
        in_2,
        in_1,
        in_0,
        normalized,
        batch_size,
        seq_len,
        hidden_dim,
        eps,
    )
    
    # Step 2: Use PyTorch's efficient transpose (view operation - essentially free)
    output = normalized.transpose(-1, -2)
    
    return output

def replacement_func():
    return fused_layernorm_transpose