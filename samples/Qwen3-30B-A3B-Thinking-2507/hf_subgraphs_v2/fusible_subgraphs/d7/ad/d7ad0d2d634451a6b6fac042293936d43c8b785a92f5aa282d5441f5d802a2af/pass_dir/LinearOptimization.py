import torch
import triton
import triton.language as tl


def pattern(in_2, in_1, in_0):
    return torch.nn.functional.linear(in_2, in_1, in_0)

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    batch,
    seq_len,
    in_features,
    out_features,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    block_m = pid_m * BLOCK_M
    block_n = pid_n * BLOCK_N
    
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    
    m_mask = block_m + m_offsets < batch * seq_len
    n_mask = block_n + n_offsets < out_features
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, in_features, BLOCK_K):
        k_end = min(k + BLOCK_K, in_features)
        k_mask = k + tl.arange(0, BLOCK_K) < k_end
        
        x = tl.load(
            x_ptr + (block_m + m_offsets[:, None]) * in_features + (k + tl.arange(0, BLOCK_K)[None, :]),
            mask=(m_mask[:, None] & k_mask[None, :]),
            other=0.0
        )
        
        w = tl.load(
            w_ptr + (k + tl.arange(0, BLOCK_K)[:, None]) * out_features + (block_n + n_offsets)[None, :],
            mask=(k_mask[:, None] & n_mask[None, :]),
            other=0.0
        )
        
        acc += tl.dot(x, w)
    
    bias = tl.load(bias_ptr + block_n + n_offsets, mask=n_mask, other=0.0)
    acc += bias
    
    tl.store(
        out_ptr + (block_m + m_offsets[:, None]) * out_features + (block_n + n_offsets)[None, :],
        acc,
        mask=(m_mask[:, None] & n_mask[None, :])
    )

@torch.fx.wrap
def optimized_linear(in_2, in_1, in_0):
    batch, seq_len, in_features = in_2.shape
    out_features = in_1.shape[0]
    
    out = torch.empty((batch, seq_len, out_features), dtype=in_2.dtype, device=in_2.device)
    
    # Constants for tiling
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    # Calculate grid dimensions
    grid_m = (batch * seq_len + BLOCK_M - 1) // BLOCK_M
    grid_n = (out_features + BLOCK_N - 1) // BLOCK_N
    
    linear_kernel[(grid_m, grid_n)](
        in_2,
        in_1,
        in_0,
        out,
        batch,
        seq_len,
        in_features,
        out_features,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )
    
    return out

def replacement_func():
    return optimized_linear