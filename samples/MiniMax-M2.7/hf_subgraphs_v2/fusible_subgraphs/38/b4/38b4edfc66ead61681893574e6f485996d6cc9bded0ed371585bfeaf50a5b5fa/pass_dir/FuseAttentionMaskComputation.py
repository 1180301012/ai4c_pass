import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire attention mask computation
def pattern(in_0):
    """ 
    Match the attention mask computation pattern from model.py
    This creates a causal attention mask by combining:
    1. Lower triangular mask creation
    2. Input mask expansion and processing
    3. Final masked_fill operations
    """
    # Create lower triangular mask
    tmp_1 = torch.full((9, 9), -3.4028234663852886e+38, device='cuda')
    tmp_2 = torch.arange(9, device='cuda')
    tmp_3 = tmp_2 + 1
    tmp_4 = tmp_3.view(9, 1)
    tmp_5 = tmp_2 < tmp_4
    tmp_6 = tmp_1.masked_fill_(tmp_5, 0)
    tmp_7 = tmp_6.to(torch.float32)
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, 9, 9)
    
    # Process input
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 9, 9)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_13 = torch.tensor(1.0, dtype=torch.float32)
    tmp_14 = tmp_13 - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to('cuda')
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    
    return tmp_19


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def attention_mask_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused attention mask computation kernel.
    Computes causal mask = (row_idx >= col_idx) ? input_mask : -inf
    Input shape: [1, N]
    Output shape: [1, 1, N, N]
    """
    # Get program ID and calculate offsets for 2D grid
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    col_idx = tl.program_id(2)
    
    # Calculate global indices
    out_offset = batch_idx * N * N + row_idx * N + col_idx
    mask = out_offset < n_elements
    
    # Load input mask value (broadcasting over columns)
    in_offset = batch_idx * N + row_idx
    in_mask = in_offset < (N * 1)  # input is [1, N]
    
    # For attention mask: element is 0 if row < col (upper triangular)
    # and equals input value otherwise (lower triangular including diagonal)
    col_offset = batch_idx * N + col_idx
    input_val = tl.load(in_ptr + col_offset, mask=in_mask, other=0.0)
    
    # Create causal mask: row_idx >= col_idx means lower triangular
    # If row < col (upper triangle), set to -inf
    # If row >= col (lower triangle), use (1 - input)
    neg_inf = float("-inf")
    one = 1.0
    
    # Determine if this position is in upper triangle (mask = 1 means -inf)
    in_upper_tri = row_idx < col_idx
    
    # Calculate output value
    # upper triangle: -inf
    # lower triangle: 1 - input (clamped to -inf if input > 1)
    input_complement = one - input_val
    output_val = tl.where(in_upper_tri, neg_inf, input_complement)
    
    # Store result
    tl.store(out_ptr + out_offset, output_val, mask=mask)


@torch.fx.wrap
def triton_attention_mask(in_0):
    """
    Fused attention mask computation using Triton.
    Input: [1, N] tensor (attention mask)
    Output: [1, 1, N, N] tensor (causal attention mask)
    """
    N = in_0.shape[1]
    batch_size = in_0.shape[0]
    
    # Output shape: [batch, 1, N, N]
    output = torch.empty((batch_size, 1, N, N), dtype=torch.float32, device='cuda')
    
    # Grid: [batch, N, N] - each program handles one output element
    grid = (batch_size, N, N)
    n_elements = batch_size * N * N
    
    BLOCK_SIZE = 1024
    
    attention_mask_kernel[grid](
        in_ptr=in_0,
        out_ptr=output,
        n_elements=n_elements,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return triton_attention_mask