import torch
import triton
import triton.language as tl

# Pattern matching for linear operation

def pattern(x, w):
    return torch.nn.functional.linear(x, w, None)

# Extract arguments for replacement

def replacement_args(x, w):
    return (x, w)

# Triton kernel for matrix multiplication (linear without bias)
@triton.jit
def triton_linear_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n,
    m,
    k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    block_n = tl.program_id(0)
    block_m = tl.program_id(1)
    
    block_n_start = block_n * BLOCK_SIZE_N
    block_m_start = block_m * BLOCK_SIZE_M

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_SIZE_K):
        # Load x: [BLOCK_SIZE_N, BLOCK_SIZE_K]
        n_mask = (block_n_start + tl.arange(0, BLOCK_SIZE_N)) < n
        k_mask = (k_start + tl.arange(0, BLOCK_SIZE_K)) < k
        mask = n_mask[:, None] & k_mask[None, :]
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(n, k),
            strides=(k, 1),
            offsets=(block_n_start, k_start),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
            order=(0, 1)
        )
        x = tl.load(x_block_ptr, boundary_check=(0, 1))
        
        # Load w: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        n_mask = (block_m_start + tl.arange(0, BLOCK_SIZE_M)) < m
        k_mask = (k_start + tl.arange(0, BLOCK_SIZE_K)) < k
        mask = n_mask[:, None] & k_mask[None, :]
        w_block_ptr = tl.make_block_ptr(
            base=w_ptr,
            shape=(m, k),
            strides=(k, 1),
            offsets=(block_m_start, k_start),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M),
            order=(1, 0)
        )
        w = tl.load(w_block_ptr, boundary_check=(0, 1))
        
        # Compute dot product with transpose_b=True
        acc += tl.dot(x, w)

    # Convert to output dtype
    acc = acc.to(tl.float16)
    
    # Compute output offsets
    offsets = (block_n_start + tl.arange(0, BLOCK_SIZE_N))[:, None] * m + \
              (block_m_start + tl.arange(0, BLOCK_SIZE_M))[None, :]
    
    # Mask for valid elements
    n_mask = (block_n_start + tl.arange(0, BLOCK_SIZE_N)) < n
    m_mask = (block_m_start + tl.arange(0, BLOCK_SIZE_M)) < m
    mask = n_mask[:, None] & m_mask[None, :]

    tl.store(out_ptr + offsets, acc, mask=mask)

# Kernel wrapper with proper dimension handling
@torch.fx.wrap
@torch.fx.wrap
def linear_kernel(x, w):
    # Calculate tensor dimensions
    batch_seq = x.numel() // x.shape[-1]  # n = batch * sequence
    m = w.shape[0]  # output features
    k = x.shape[-1]  # input features
    
    # Block sizes optimized for V100/A100
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 64
    
    # Grid dimensions
    grid_n = (batch_seq + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid = (grid_n, grid_m)
    
    # Allocate output
    out = torch.empty(batch_seq, m, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    triton_linear_kernel[grid](
        x, w, out,
        batch_seq, m, k,
        BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_K
    )
    
    # Reshape to original dimensions
    return out.view(*x.shape[:-1], m)

# Replacement function

def replacement_func():
    return linear_kernel