import torch
import triton
import triton.language as tl


def pattern(in_1):
    """
    Pattern for view + transpose on in_1
    Matches: in_1.view(batch, -1, num_heads, head_dim).transpose(1, 2)
    """
    tmp_0 = in_1.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def fused_view_transpose_kernel(
    in_ptr,
    out_ptr,
    batch,
    seq_len,
    num_heads,
    head_dim,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused kernel for view + transpose operation with better memory access.
    Input: [batch, seq_len, num_heads * head_dim]
    Output: [batch, num_heads, seq_len, head_dim]
    
    Each block processes BLOCK_H heads x BLOCK_D dimensions for one (batch, seq) position
    """
    # Each program handles one (batch, seq) position
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    b = pid_batch
    s = pid_seq
    
    # Process all heads and dimensions for this (b, s) position
    for h in range(num_heads):
        # Load contiguous chunk from input: [seq_len, num_heads * head_dim]
        # Input offset: b * (seq_len * num_heads * head_dim) + s * (num_heads * head_dim) + h * head_dim
        in_base = b * (seq_len * num_heads * head_dim) + s * (num_heads * head_dim) + h * head_dim
        
        # Vectorized load
        d_offsets = tl.arange(0, BLOCK_D)
        mask = d_offsets < head_dim
        in_offsets = in_base + d_offsets
        data = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
        
        # Store to output: [batch, num_heads, seq_len, head_dim]
        out_base = b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim) + s * head_dim
        out_offsets = out_base + d_offsets
        tl.store(out_ptr + out_offsets, data, mask=mask)


@torch.fx.wrap
def fused_view_transpose(in_tensor, batch, seq_len, num_heads, head_dim):
    """
    Optimized fused view + transpose operation.
    """
    # Output shape: [batch, num_heads, seq_len, head_dim]
    out = torch.empty(batch, num_heads, seq_len, head_dim, 
                      dtype=in_tensor.dtype, device=in_tensor.device)
    
    BLOCK_D = 64  # Process 64 elements at a time (matches head_dim typically)
    grid = (batch, seq_len)
    
    fused_view_transpose_kernel[grid](
        in_tensor,
        out,
        batch,
        seq_len,
        num_heads,
        head_dim,
        BLOCK_H=1,
        BLOCK_D=BLOCK_D,
    )
    
    return out


def replacement_func():
    def optimized(in_1):
        # Extract shape information
        batch = 32
        seq_len = in_1.shape[1]
        num_heads = 1
        head_dim = 64
        return fused_view_transpose(in_1, batch, seq_len, num_heads, head_dim)
    
    return optimized