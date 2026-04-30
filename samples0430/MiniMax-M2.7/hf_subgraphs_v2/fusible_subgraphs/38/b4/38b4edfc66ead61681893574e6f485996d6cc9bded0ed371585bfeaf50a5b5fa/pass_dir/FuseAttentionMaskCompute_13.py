import torch
import triton
import triton.language as tl

# Pattern matching function for attention mask computation (size 13)
def pattern(in_0):
    # Create lower triangular mask base (-inf everywhere)
    tmp_1 = torch.full((13, 13), -3.4028234663852886e+38, device=torch.device(type='cuda', index=0))
    
    # Create indices for lower triangular comparison
    tmp_2 = torch.arange(13, device=torch.device(type='cuda', index=0))
    tmp_3 = tmp_2 + 1
    tmp_4 = tmp_3.view(13, 1)
    
    # Create lower triangular boolean mask
    tmp_5 = tmp_2 < tmp_4
    
    # Fill lower triangle with 0
    tmp_6 = tmp_1.masked_fill_(tmp_5, 0)
    
    # Convert to float32 and add dimensions
    tmp_7 = tmp_6.to(torch.float32)
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, 13, 13)
    
    # Process input tensor: expand to 4D and convert
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 13, 13)
    tmp_12 = tmp_11.to(torch.float32)
    
    # Create inverted mask (1 - input)
    tmp_13 = torch.tensor(1.0, dtype=torch.float32)
    tmp_14 = tmp_13 - tmp_12
    
    # Convert to boolean
    tmp_15 = tmp_14.to(torch.bool)
    
    # Fill -inf where inverted input is 1
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    
    # Move to GPU and convert to bool
    tmp_17 = tmp_16.to(torch.device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    
    # Apply mask to lower triangular mask
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    
    return tmp_19

def replacement_args(in_0):
    return (in_0,)

# Autotune configuration for different block sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def attention_mask_kernel_13(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate offsets for the NxN output [1, 1, N, N]
    offs_i = tl.arange(0, BLOCK_SIZE)
    offs_j = tl.arange(0, BLOCK_SIZE)
    
    # Input mask value (broadcast from [1, N] to [1, 1, N, N] across j dimension)
    mask_idx = offs_j % N
    
    # Load the input attention mask value (int64)
    in_val = tl.load(in_ptr + mask_idx).to(tl.float32)
    
    # Calculate inverted input: 1.0 - in_val
    # This gives: 0 when input was 1, 1 when input was 0
    inv_mask_val = 1.0 - in_val
    
    # Convert to bool - True when inverted value is non-zero (original was 0)
    neg_inf = float("-1e38")
    should_mask_from_input = inv_mask_val != 0.0
    
    # Create lower triangular mask: fill -inf for upper triangle (i < j)
    i_idx = offs_i[:, None]
    j_idx = offs_j[None, :]
    upper_triangle = i_idx < j_idx
    
    # Combine masks: fill with -inf if either condition is true
    should_fill = upper_triangle | should_mask_from_input
    
    # Build output: 0 for valid positions, -inf for masked positions
    result = tl.where(should_fill, neg_inf, 0.0)
    
    # Store output [1, 1, N, N] - row-major
    row_offsets = offs_i[:, None] * N
    col_offsets = offs_j[None, :]
    out_offset = row_offsets + col_offsets
    tl.store(out_ptr + out_offset, result)


@torch.fx.wrap
def attention_mask_wrapper_13(in_0):
    N = 13  # Fixed size for this pass
    BLOCK_SIZE = 32
    
    # Output shape is [1, 1, N, N]
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device='cuda')
    
    # Grid: single kernel launch
    grid = (1,)
    
    attention_mask_kernel_13[grid](
        in_ptr=in_0,
        out_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return attention_mask_wrapper_13