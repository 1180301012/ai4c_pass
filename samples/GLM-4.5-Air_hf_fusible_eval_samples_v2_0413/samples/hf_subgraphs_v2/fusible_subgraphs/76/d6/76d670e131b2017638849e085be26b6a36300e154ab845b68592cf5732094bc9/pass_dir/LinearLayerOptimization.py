import torch
import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_cols_out,
    n_cols_in,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID - each program handles one (M,N) output element
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets for this program
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Create masks for all dimensions
    m_mask = m_offset + tl.arange(0, BLOCK_SIZE_M) < n_rows
    n_mask = n_offset + tl.arange(0, BLOCK_SIZE_N) < n_cols_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension (inner dimension)
    for k in range(0, n_cols_in, BLOCK_SIZE_K):
        k_offset = k
        k_mask = k_offset + tl.arange(0, BLOCK_SIZE_K) < n_cols_in
        
        # Load input block: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        x_ptr_base = x_ptr + (m_offset + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)[:, None] * n_cols_in
        x_ptrs = x_ptr_base + tl.arange(0, BLOCK_SIZE_K)[None, :]
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        
        # Load weight block: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        weight_ptr_base = weight_ptr + (k_offset + tl.arange(0, BLOCK_SIZE_K)).to(tl.int64)[:, None] * n_cols_out
        weight_ptrs = weight_ptr_base + tl.arange(0, BLOCK_SIZE_N)[None, :]
        weight = tl.load(weight_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        
        # Matrix multiplication
        acc += tl.dot(x, weight)
    
    # Load bias for this N block
    bias_ptrs = bias_ptr + n_offset + tl.arange(0, BLOCK_SIZE_N)
    bias = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    
    # Add bias and convert back to original dtype
    acc = acc + bias[None, :]
    out = acc.to(tl.float16 if x_ptr.dtype.element_ty == tl.float16 else tl.bfloat16)
    
    # Store result
    out_ptr_base = out_ptr + (m_offset + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)[:, None] * n_cols_out
    out_ptrs = out_ptr_base + n_offset + tl.arange(0, BLOCK_SIZE_N)[None, :]
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def optimized_linear(x, weight, bias):
    # Get input/output shapes
    n_rows = x.shape[0]
    n_cols_in = x.shape[1]
    n_cols_out = weight.shape[0]
    
    # Optimize block sizes for small batch sizes to avoid shared memory overflow
    if n_rows >= 128:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 128, 32
    elif n_rows >= 32:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 128, 32
    else:  # Small batch sizes (1, 32)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 64, 32
    
    # Calculate grid size
    num_blocks_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (n_cols_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Allocate output buffer
    out = torch.empty((n_rows, n_cols_out), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    linear_kernel[(num_blocks_m, num_blocks_n)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_rows=n_rows,
        n_cols_out=n_cols_out,
        n_cols_in=n_cols_in,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def pattern(in_6, in_5, in_4):
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    return linear

def replacement_args(in_6, in_5, in_4):
    return (in_6, in_5, in_4)

def replacement_func():
    return optimized_linear