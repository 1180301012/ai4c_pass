import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern matching the position ID generation:
    ne(1) -> int() -> cumsum(dim=1) -> type_as -> add(0) -> mul(mask) -> long() -> add(1)
    """
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_position_id_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for position ID generation.
    Each program handles one row (batch element).
    Optimized with autotune for different sequence lengths.
    """
    row_idx = tl.program_id(0)
    
    row_start = row_idx * seq_len
    
    # Load entire row
    offsets = tl.arange(0, BLOCK_SIZE)
    load_mask = offsets < seq_len
    input_offsets = row_start + offsets
    input_vals = tl.load(input_ptr + input_offsets, mask=load_mask, other=0)
    
    # Compute: ne(1).int()
    mask_vals = (input_vals != 1).to(tl.int32)
    
    # Compute cumsum using prefix sum
    # cumsum[i] = sum of mask_vals[0:i+1]
    cumsum_vals = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Build cumsum by computing prefix sums
    # For position i, sum all elements from 0 to i (inclusive)
    # Optimized: skip positions beyond seq_len
    for i in range(BLOCK_SIZE):
        # Skip if this position is beyond sequence length
        # This reduces work when BLOCK_SIZE > seq_len
        if i < seq_len:
            # Create mask for elements 0..i
            prefix_mask = (offsets <= i) & load_mask
            # Compute sum of masked values
            prefix_sum = tl.sum(tl.where(prefix_mask, mask_vals, 0))
            # Store at position i
            cumsum_vals = tl.where(offsets == i, prefix_sum, cumsum_vals)
    
    # Compute result: cumsum * mask + 1
    result = cumsum_vals * mask_vals + 1
    
    # Store as int64
    tl.store(output_ptr + input_offsets, result.to(tl.int64), mask=load_mask)


@torch.fx.wrap
def fused_position_id_generation(input_tensor):
    """
    Optimized position ID generation using a single Triton kernel.
    Uses autotune to select the best BLOCK_SIZE for each sequence length.
    """
    batch_size, seq_len = input_tensor.shape
    output = torch.empty_like(input_tensor, dtype=torch.int64)
    
    grid = (batch_size,)
    
    # Autotune will automatically select the best BLOCK_SIZE
    fused_position_id_kernel[grid](
        input_tensor,
        output,
        batch_size,
        seq_len,
    )
    
    return output


def replacement_func():
    return fused_position_id_generation