import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Linear transformation
    tmp = torch.nn.functional.linear(x, weight, bias)
    # Reshape for multi-head attention - avoid * unpacking which causes Proxy iteration issue
    tmp_view = tmp.view(x.shape[0], x.shape[1], -1, 64)
    # Transpose for attention: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
    tmp_transpose = tmp_view.transpose(1, 2)
    return tmp_transpose

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def linear_view_transpose_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    num_heads,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute grid positions
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.program_id(2)
    
    # Create offsets for matrix multiplication
    x_offsets = m * BLOCK_SIZE_M * hidden_size + k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    w_offsets = n * BLOCK_SIZE_N + k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    out_offsets = m * BLOCK_SIZE_M * num_heads * 64 + n * BLOCK_SIZE_N + k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Load data
    x = tl.load(x_ptr + x_offsets, mask=k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < hidden_size, other=0.0)
    w = tl.load(weight_ptr + w_offsets, mask=k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < hidden_size, other=0.0)
    
    # Matrix multiplication
    acc = tl.dot(x, w)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + n, other=0.0)
        acc += bias_val
    
    # Store result in transposed format for attention
    tl.store(out_ptr + out_offsets, acc, mask=k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < 64)

@torch.fx.wrap
def linear_view_transpose_fused(x, weight, bias):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    hidden_size = x.shape[2]
    num_heads = hidden_size // 64 if hidden_size % 64 == 0 else hidden_size // 64
    head_dim = 64
    
    # Calculate output shape for transposed format
    out_shape = (batch_size, num_heads, seq_len, head_dim)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Determine grid size
    BLOCK_M = 1
    BLOCK_N = 128
    BLOCK_K = 64
    
    num_blocks_m = (batch_size * seq_len + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (num_heads * head_dim + BLOCK_N - 1) // BLOCK_N
    num_blocks_k = (hidden_size + BLOCK_K - 1) // BLOCK_K
    
    # Launch kernel
    linear_view_transpose_kernel[(num_blocks_m, num_blocks_n, num_blocks_k)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )
    
    return out

def replacement_func():
    return linear_view_transpose_fused