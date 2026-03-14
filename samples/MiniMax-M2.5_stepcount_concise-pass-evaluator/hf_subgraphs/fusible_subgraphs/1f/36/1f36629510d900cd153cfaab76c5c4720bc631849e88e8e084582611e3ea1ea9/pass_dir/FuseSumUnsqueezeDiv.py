import torch
import triton
import triton.language as tl


# Pattern matching function
# Let's try a very simple pattern first - just match sum + unsqueeze
def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel that fuses sum + unsqueeze + divide
@triton.jit
def fused_sum_div_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    H: tl.constexpr,  # 196 - the dim we sum over
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. Sum along last dimension
    2. Divide input by sum (with broadcasting)
    """
    # Each program processes a contiguous block of (batch, head, seq_pos) positions
    # For each such position, we need to compute sum over the last dim and divide
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute the sum for this block
    # We need to sum across H elements for each position
    # Since each thread has one element, we need a reduction
    
    # For correct reduction, let's use a different approach:
    # Each position has H=196 elements in the last dim
    # We need to compute sum for each position
    
    # Actually, let's rethink this - the kernel structure needs to match the tensor layout
    # Input shape: [1, 16, 196, 196] -> flattened view for processing
    
    # We'll use a per-row reduction approach
    # For each (b,h,seq) position, sum the H values and divide
    
    # Let's compute the flat indices properly
    pass


# Simplified approach: use a more straightforward kernel
@triton.jit
def fused_sum_div_kernel_v2(
    input_ptr,
    output_ptr,
    n_elements,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: compute sum along dim=-1 and divide input by sum"""
    # pid = block index
    pid = tl.program_id(0)
    
    # Each block processes one (b, h, seq) position
    # There are 1*16*196 = 3136 such positions
    # Each position has H=196 elements in the last dimension
    
    # Load all H elements for this position
    # H is the reduction dimension (196)
    
    # Create offsets for the H elements at this position
    # Position index: pid
    # Element indices within H: 0..H-1
    
    # Actually, we need multiple blocks to cover all elements
    # Let's do a different decomposition:
    # Each block handles multiple positions with reduction
    
    # Simpler: each block handles one position and loops over H elements
    block_start = pid * BLOCK_SIZE
    
    # Sum reduction over H elements
    sum_val = 0.0
    for i in range(H):
        idx = block_start + i
        if idx < n_elements:
            val = tl.load(input_ptr + idx)
            sum_val += val
    
    # Broadcast sum to all elements in this position
    # Now divide each element by the sum
    for i in range(H):
        idx = block_start + i
        if idx < n_elements:
            val = tl.load(input_ptr + idx)
            result = val / (sum_val + 1e-8)  # Add small epsilon for stability
            tl.store(output_ptr + idx, result)


# Better approach: Use block-level parallelization more efficiently
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_sum_div_kernel_v3(
    input_ptr,
    output_ptr,
    n_elements,
    H: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized fused kernel: compute sum along dim=-1 and divide input by sum
    
    Grid: (num_positions, ) where num_positions = n_elements // H
    Each block processes BLOCK_SIZE_M positions, each with BLOCK_SIZE_N reduction
    """
    # Row position (b*16*196 + h*196 + seq)
    row_idx = tl.program_id(0) * BLOCK_SIZE_M
    col_idx = tl.program_id(1) * BLOCK_SIZE_N
    
    # Base offset for this position
    row_offset = row_idx * H
    col_offset = col_idx
    
    # Initialize sum accumulator
    sum_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Reduction loop over H dimension
    for h in range(H):
        offsets = row_offset + h * tl.arange(0, BLOCK_SIZE_M)[:, None] * H + col_offset + tl.arange(0, BLOCK_SIZE_N)[None, :]
        mask = (row_offset + h * tl.arange(0, BLOCK_SIZE_M)[:, None] * H + col_offset + tl.arange(0, BLOCK_SIZE_N)[None, :]) < n_elements
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        sum_acc += vals
    
    # Horizontal sum to get total per position
    # sum_acc now has shape (BLOCK_SIZE_M, BLOCK_SIZE_N), sum along N dim for each M
    total_sum = tl.sum(sum_acc, axis=1)[:, None]
    
    # Now load and divide
    for h in range(H):
        offsets = row_offset + h * tl.arange(0, BLOCK_SIZE_M)[:, None] * H + col_offset + tl.arange(0, BLOCK_SIZE_N)[None, :]
        mask = (row_offset + h * tl.arange(0, BLOCK_SIZE_M)[:, None] * H + col_offset + tl.arange(0, BLOCK_SIZE_N)[None, :]) < n_elements
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        result = vals / (total_sum + 1e-8)
        tl.store(output_ptr + offsets, result, mask=mask)


# Let's simplify: each position (b,h,seq) is independent
# For shape [1, 16, 196, 196], we have 1*16*196 = 3136 positions
# Each position has 196 elements to sum

@triton.jit
def fused_sum_div_kernel_simple(
    input_ptr,
    output_ptr,
    num_positions,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program (block) processes one position (b,h,seq)
    There are num_positions = 1*16*196 = 3136 programs
    Each block sums H=196 elements and divides each by the sum
    """
    # Each block handles one (b,h,seq) position
    pid = tl.program_id(0)
    
    if pid >= num_positions:
        return
    
    # Compute base offset for this position
    base_offset = pid * H
    
    # First, compute the sum across H elements
    sum_val = 0.0
    for i in range(H):
        offset = base_offset + i
        val = tl.load(input_ptr + offset)
        sum_val += val
    
    # Add epsilon for numerical stability
    sum_val = sum_val + 1e-8
    
    # Now divide each element by sum and store
    for i in range(H):
        offset = base_offset + i
        val = tl.load(input_ptr + offset)
        result = val / sum_val
        tl.store(output_ptr + offset, result)


@torch.fx.wrap
def fused_sum_div_wrapper(in_0):
    """
    Wrapper function that launches the fused sum + divide kernel.
    Input shape: [1, 16, 196, 196]
    Computes: output[b,h,seq,:] = input[b,h,seq,:] / sum(input[b,h,seq,:])
    """
    # Input shape is [B, H, S, S] = [1, 16, 196, 196]
    B, H, S, S2 = in_0.shape
    assert S == S2  # Square last two dims
    
    num_positions = B * H * S  # 1 * 16 * 196 = 3136
    H_dim = S  # 196
    
    # Flatten input to [num_positions, H_dim]
    input_flat = in_0.view(-1)
    
    # Output tensor
    output_flat = torch.empty_like(input_flat)
    
    # Launch kernel: one block per position
    BLOCK_SIZE = 256  # Enough for H=196 elements per position
    
    fused_sum_div_kernel_simple[(num_positions,)](
        input_ptr=input_flat,
        output_ptr=output_flat,
        num_positions=num_positions,
        H=H_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back
    output = output_flat.view(B, H, S, S2)
    
    return output


def replacement_func():
    return fused_sum_div_wrapper