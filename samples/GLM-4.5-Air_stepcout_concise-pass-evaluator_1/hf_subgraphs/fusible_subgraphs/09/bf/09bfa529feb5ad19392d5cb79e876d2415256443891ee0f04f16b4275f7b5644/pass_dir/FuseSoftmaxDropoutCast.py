import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Match the exact computation pattern from the graphs
    # In the graphs, dropout_p is fixed to either 0.1 or 0.0, so we match common case
    tmp_0 = x + y
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    tmp_3 = tmp_2.to(torch.float32)
    return (tmp_3,)

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_attention_kernel(
    x_ptr, y_ptr, out_ptr,
    batch_size, num_heads, seq_len, seq_len2,
    dropout_p: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Calculate program IDs
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute memory addresses for this program
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Create broadcastable offsets
    m_offsets = m_offsets[:, None]
    n_offsets = n_offsets[None, :]
    
    # Calculate total offsets (batch, head, m, n)
    offsets = pid_b * num_heads * seq_len * seq_len + pid_h * seq_len * seq_len + m_offsets * seq_len + n_offsets
    
    # Load inputs with bounds checking
    mask = (m_offsets < seq_len) & (n_offsets < seq_len2)
    mask = mask.to(tl.int32)
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Element-wise addition
    sum_val = x + y
    
    # Softmax along the last dimension (n dimension)
    max_val = tl.max(sum_val, axis=1, keepdim=True)
    exp_val = tl.exp(sum_val - max_val)
    sum_exp = tl.sum(exp_val, axis=1, keepdim=True)
    softmax_val = exp_val / (sum_exp + 1e-8)  # Add small epsilon for stability
    
    # Dropout (inference mode, so just pass through)
    if dropout_p > 0.0 and dropout_p < 1.0:
        # For inference, dropout with train=False should not change values
        out_val = softmax_val
    else:
        out_val = softmax_val
    
    # Store the result (already float32 from softmax)
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_attention_forward(x, y, dropout_p=0.1):
    # Get tensor shapes
    batch_size, num_heads, seq_len, seq_len2 = x.shape
    
    # Output tensor (already float32)
    out = torch.empty_like(x, dtype=torch.float32)
    
    # Choose block sizes based on sequence length for good occupancy
    if seq_len <= 64:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
    elif seq_len <= 128:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
    elif seq_len <= 256:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
    else:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
    
    # Calculate grid dimensions
    num_blocks_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid = (num_blocks_m, num_heads, batch_size)
    
    # Launch the kernel
    fused_attention_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        seq_len2=seq_len2,
        dropout_p=dropout_p,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return (out,)

def replacement_func():
    return fused_attention_forward