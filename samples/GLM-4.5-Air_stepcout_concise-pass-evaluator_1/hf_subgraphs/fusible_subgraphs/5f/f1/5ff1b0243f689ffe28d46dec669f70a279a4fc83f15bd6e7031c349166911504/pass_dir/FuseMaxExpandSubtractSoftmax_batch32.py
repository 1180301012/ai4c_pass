import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_softmax_kernel_batch32(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    stride_input_batch,
    stride_input_seq,
    stride_input_last,
    stride_output_batch,
    stride_output_seq,
    stride_output_last,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for numerically stable softmax on 3D tensor.
    Input shape: [batch_size, seq_len, seq_len]
    Softmax is computed along the last dimension (dim=-1).
    
    Each program processes one row: input[batch_idx, seq_idx, :]
    """
    # Each block processes one row: input[batch_idx, seq_idx, :]
    # Total number of rows = batch_size * seq_len
    row_id = tl.program_id(0)
    batch_idx = row_id // seq_len
    seq_idx = row_id % seq_len
    
    if batch_idx >= batch_size:
        return
    
    # Calculate the base offset for this row
    # input[batch_idx, seq_idx, :] is at offset = batch_idx * stride_input_batch + seq_idx * stride_input_seq
    row_base = batch_idx * stride_input_batch + seq_idx * stride_input_seq
    
    # Create offsets for the last dimension
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < seq_len
    
    # Load the entire row
    row_ptrs = input_ptr + row_base + col_offsets * stride_input_last
    x = tl.load(row_ptrs, mask=mask, other=float('-inf'))
    
    # Step 1: Compute max over the row
    max_val = tl.max(x, axis=0)
    
    # Step 2: Compute exp(x - max) with numerical stability
    max_val = tl.broadcast_to(max_val, (BLOCK_SIZE,))
    exp_val = tl.exp(x - max_val)
    exp_val = exp_val * mask  # Zero out masked positions
    
    # Step 3: Compute sum of exponentials
    sum_exp = tl.sum(exp_val, axis=0)
    
    # Step 4: Normalize
    softmax_vals = exp_val / sum_exp
    
    # Store result
    out_row_base = batch_idx * stride_output_batch + seq_idx * stride_output_seq
    out_row_ptrs = output_ptr + out_row_base + col_offsets * stride_output_last
    tl.store(out_row_ptrs, softmax_vals, mask=mask)


@torch.fx.wrap
def fused_softmax_wrapper_batch32(in_0, in_1):
    """
    Fused numerically stable softmax kernel for batch_size=32.
    
    This fuses: max -> expand -> subtract -> softmax
    into a single kernel for better performance.
    
    Args:
        in_0: Input tensor of shape [32, seq_len, seq_len]
        in_1: Input tensor to be reshaped [32, 512, 64, 64]
    
    Returns:
        Tuple of (softmax_result, reshaped_in_1)
    """
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]
    
    # Allocate output
    output = torch.empty_like(in_0)
    
    # Configure block size based on seq_len
    BLOCK_SIZE = 1024
    if seq_len <= 512:
        BLOCK_SIZE = 512
    elif seq_len <= 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    # Launch kernel - one program per row (batch_size * seq_len total rows)
    grid = (batch_size * seq_len,)
    
    fused_softmax_kernel_batch32[grid](
        in_0,
        output,
        batch_size,
        seq_len,
        in_0.stride(0),  # stride for batch dimension
        in_0.stride(1),  # stride for seq dimension  
        in_0.stride(2),  # stride for last dimension
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SIZE,
    )
    
    # Handle view on in_1 for batch_size=32
    # Original: in_1.view(32, 512, -1)
    in_1_reshaped = in_1.view(32, 512, -1)
    
    return output, in_1_reshaped


def pattern(in_0, in_1):
    """
    Pattern matching numerically stable softmax for batch_size=32.
    Must return BOTH outputs to match the model's return structure.
    """
    # Compute max along last dimension
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]  # Extract max values
    
    # Broadcast max to input shape
    tmp_2 = tmp_1.expand_as(in_0)
    
    # Subtract (numerical stability)
    tmp_3 = tmp_2 - in_0
    
    # Apply softmax
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    
    # View on in_1 for batch_size=32 - MUST include this even though we don't need to optimize it
    tmp_5 = in_1.view(32, 512, -1)
    
    return tmp_4, tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_softmax_wrapper_batch32