import torch
import triton
import triton.language as tl


# Pattern: view(1, seq, 16, 64) -> transpose(1, 2) -> reshape(16, seq, 64)
# Fused into a single optimized Triton kernel
def pattern(in_view):
    tmp_view = in_view.view(1, -1, 16, 64)
    tmp_trans = tmp_view.transpose(1, 2)
    tmp_reshape = tmp_trans.reshape(16, -1, 64)
    return tmp_reshape


def pattern_query(in_view):
    tmp_view = in_view.view(1, 1, 16, 64)
    tmp_trans = tmp_view.transpose(1, 2)
    tmp_reshape = tmp_trans.reshape(16, -1, 64)
    return tmp_reshape


def pattern_transpose(in_tensor):
    return in_tensor.transpose(1, 2)


def replacement_args(in_tensor):
    return (in_tensor,)


# Optimized Triton kernel for attention reshape
@triton.jit
def fused_reshape_kernel(
    input_ptr,
    output_ptr,
    seq_len,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute output indices
    # Transform from (1, seq, 16*64) -> (16, seq, 64)
    seq_idx = offsets // (16 * 64)
    elem_idx = offsets % (16 * 64)
    head_idx = elem_idx // 64
    inner_idx = elem_idx % 64
    
    # Output position
    out_offsets = (head_idx * seq_len + seq_idx) * 64 + inner_idx
    
    tl.store(output_ptr + out_offsets, input, mask=mask)


@torch.fx.wrap
def fused_reshape_wrapper(in_tensor):
    """Highly optimized fused kernel for attention reshape pattern."""
    # Input: (1, seq, 1024)
    # Output: (16, seq, 64)
    
    # Handle both (1, seq, 1024) and (seq, 1024) inputs
    if in_tensor.dim() == 3 and in_tensor.shape[0] == 1:
        input = in_tensor.squeeze(0)  # (seq, 1024)
    else:
        input = in_tensor
    
    seq_len = input.shape[0]
    total_elements = seq_len * 16 * 64
    
    output = torch.empty((16, seq_len, 64), device=input.device, dtype=input.dtype)
    
    # Launch kernel with fixed block size
    BLOCK_SIZE = 512
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_reshape_kernel[(num_programs,)](
        input_ptr=input,
        output_ptr=output,
        seq_len=seq_len,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_reshape_wrapper