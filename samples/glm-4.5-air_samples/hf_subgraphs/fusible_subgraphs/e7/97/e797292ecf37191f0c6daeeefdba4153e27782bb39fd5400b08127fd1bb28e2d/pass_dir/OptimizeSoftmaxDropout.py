import torch
import triton
import triton.language as tl

def pattern(attention_weights, _):
    """
    Pattern matching for softmax + dropout(p=0.0)
    Since dropout with p=0.0 is identity, this is equivalent to just softmax
    """
    softmax_out = torch.nn.functional.softmax(attention_weights, dim=-1)
    dropout_out = torch.nn.functional.dropout(softmax_out, p=0.0, training=False)
    return dropout_out

def replacement_args(attention_weights, _):
    """
    Extract arguments for replacement - just need the attention weights
    """
    return (attention_weights,)

@triton.heuristics({
    "BLOCK_SIZE": lambda args: [1024, 2048, 4096, 8192],
})
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized softmax kernel with better memory coalescing and occupancy
    """
    # Each program processes a whole row for softmax
    row_idx = tl.program_id(0)
    offset = row_idx * n_elements + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load the row
    x = tl.load(input_ptr + offset, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x, mask=mask)
    tl.store(output_ptr + row_idx * n_elements + 0, max_val)
    
    # Compute exponential values
    exp_x = tl.exp(x - max_val)
    
    # Compute sum for normalization
    sum_exp = tl.sum(exp_x, mask=mask)
    
    # Normalize to get softmax
    softmax_out = exp_x / sum_exp
    
    # Store the result
    tl.store(output_ptr + offset, softmax_out, mask=mask)

@triton.heuristics({
    "BLOCK_SIZE_M": lambda args: [32, 64, 128],
    "BLOCK_SIZE_N": lambda args: [32, 64, 128, 256],
})
@triton.jit
def parallel_softmax_kernel(
    input_ptr,
    output_ptr,
    m,
    n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Alternative softmax kernel with 2D grid for better utilization
    """
    pid = tl.program_id(0)
    mid = pid // tl.cdiv(n, BLOCK_SIZE_N)
    nid = pid % tl.cdiv(n, BLOCK_SIZE_N)
    
    if mid >= m:
        return
        
    offset_row = mid * n
    offset_col = nid * BLOCK_SIZE_N
    
    m_end = min(offset_row + n, offset_row + BLOCK_SIZE_N)
    
    x = tl.load(input_ptr + offset_row + offset_col, 
                mask=offset_col + tl.arange(0, BLOCK_SIZE_N) < (m_end - offset_row),
                other=-float('inf')).to(tl.float32)
    
    max_x = tl.max(x)
    sum_x = tl.sum(tl.exp(x - max_x))
    
    tl.store(output_ptr + offset_row + offset_col, 
             (tl.exp(x - max_x) / sum_x),
             mask=offset_col + tl.arange(0, BLOCK_SIZE_N) < (m_end - offset_row))

@torch.fx.wrap
def optimized_softmax(attention_weights):
    """
    Optimized softmax implementation that removes redundant dropout
    """
    m, n = attention_weights.shape[-2:]
    n_elements = m * n
    
    # Choose kernel strategy based on tensor shape
    if n_elements > 8192:
        # Use 2D grid for large tensors
        num_blocks = (m * n_elements + 128 * 256 - 1) // (128 * 256)
        parallel_softmax_kernel[(num_blocks,)](
            attention_weights,
            attention_weights,  # in-place
            m,
            n,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=256,
        )
    else:
        # Use simple row-wise processing for smaller tensors
        num_rows = m
        softmax_kernel[(num_rows,)](
            attention_weights,
            attention_weights,  # in-place
            n_elements,
            BLOCK_SIZE=1024,
        )
    
    return attention_weights

def replacement_func():
    """
    Return the optimized softmax function
    """
    return optimized_softmax