import torch
import triton
import triton.language as tl


@triton.jit
def scale_softmax_kernel(
    input_ptr, output_ptr,
    batch_size, num_heads, seq_len,
    SCALE_FACTOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that performs: scale * softmax
    Then we use PyTorch's transpose for the final transpose
    
    Grid: (batch_size * num_heads * seq_len,)
    Each program handles one row of the input
    """
    # program_id(0) = row index in the input
    program_id = tl.program_id(0)
    
    # Calculate batch_idx, head_idx, and row_idx from program_id
    batch_head = program_id // seq_len
    row_idx = program_id % seq_len
    
    batch_idx = batch_head // num_heads
    head_idx = batch_head % num_heads
    
    # Calculate the starting offset for this row in the input tensor
    row_offset = batch_idx * num_heads * seq_len * seq_len + head_idx * seq_len * seq_len + row_idx * seq_len
    
    # Load all values in this row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    row_start = input_ptr + row_offset
    row_vals = tl.load(row_start + offsets, mask=mask, other=-float('inf'))
    
    # Step 1: Scale the values
    row_vals = row_vals * SCALE_FACTOR
    
    # Step 2: Find max for numerical stability
    max_val = tl.max(row_vals, axis=0)
    
    # Step 3: Compute exp(x - max)
    row_vals = tl.exp(row_vals - max_val)
    
    # Step 4: Compute sum
    sum_val = tl.sum(row_vals, axis=0)
    
    # Step 5: Normalize
    row_vals = row_vals / sum_val
    
    # Store to output (same layout as input)
    output_start = output_ptr + row_offset
    tl.store(output_start + offsets, row_vals, mask=mask)


def fused_scale_softmax_transpose(x: torch.Tensor) -> torch.Tensor:
    """
    Fused scaled softmax with transpose.
    Input: [..., seq_len, seq_len] 
    Output: [..., seq_len, seq_len] (transposed last two dims)
    """
    batch_size = x.shape[0]
    num_heads = x.shape[1]
    seq_len = x.shape[2]
    
    # Scale factor for the pattern (1/sqrt(32))
    SCALE_FACTOR = 0.1767766952966369
    
    # For small batch sizes, use PyTorch's optimized implementations
    # The overhead of Triton kernel launch isn't worth it for small tensors
    total_elements = batch_size * num_heads * seq_len * seq_len
    use_triton = total_elements > 1000000  # Only use Triton for larger tensors
    
    if use_triton:
        # First do scaled softmax using Triton
        scaled_softmax = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        grid = (batch_size * num_heads * seq_len,)
        
        scale_softmax_kernel[grid](
            x, scaled_softmax,
            batch_size, num_heads, seq_len,
            SCALE_FACTOR,
            BLOCK_SIZE,
        )
    else:
        # Use PyTorch for small tensors
        scaled_softmax = x * SCALE_FACTOR
        scaled_softmax = scaled_softmax.softmax(dim=-1)
    
    # Then transpose using PyTorch (this is cheap)
    output = scaled_softmax.transpose(-2, -1)
    
    return output


@torch.fx.wrap
def fused_scale_softmax_transpose_wrapper(x: torch.Tensor) -> torch.Tensor:
    return fused_scale_softmax_transpose(x)


def pattern(in_0):
    """Match the computation pattern: scale * softmax * transpose"""
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_scale_softmax_transpose_wrapper