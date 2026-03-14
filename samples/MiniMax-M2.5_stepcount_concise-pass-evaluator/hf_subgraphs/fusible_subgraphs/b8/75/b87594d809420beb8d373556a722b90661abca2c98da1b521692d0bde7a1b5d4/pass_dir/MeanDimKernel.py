import torch
import triton
import triton.language as tl


# Pattern matching function - matches the mean operation from the model
def pattern(in_3):
    """
    Match the mean operation: in_3.mean(-2)
    This computes the mean over dimension -2 (the 49 dimension).
    """
    result = in_3.mean(-2)
    return result


# Argument extraction function
def replacement_args(in_3):
    return (in_3,)


# Optimized Triton kernel for mean reduction over dim -2
@triton.autotune(
    configs=[
        # Different tile sizes for different batch sizes
        triton.Config({'BLOCK_SIZE': 1}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 4}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 8}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 16}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=1),
    ],
    key=['batch_size'],
)
@triton.jit
def triton_mean_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, hidden_size,
    stride_input_batch, stride_input_seq, stride_input_hidden,
    stride_output_batch, stride_output_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for mean reduction over dimension -2.
    Input shape: (batch_size, seq_len, hidden_size) = (batch, 49, 448)
    Output shape: (batch_size, hidden_size) = (batch, 448)
    
    Computes: output[b, h] = mean over seq of input[b, seq, h]
    """
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    # Calculate total number of elements in output
    # Each hidden dimension is processed independently
    # We process hidden_size elements in total across the grid
    
    # Compute starting position for this thread
    hidden_offset = pid_hidden * BLOCK_SIZE
    hidden_offsets = hidden_offset + tl.arange(0, BLOCK_SIZE)
    mask_hidden = hidden_offsets < hidden_size
    
    # Compute the mean over seq dimension
    # For each (batch, hidden), sum all seq values and divide by seq_len
    seq_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load and accumulate over seq dimension
    for seq_idx in range(seq_len):
        # Create offsets for this sequence position
        input_offsets = (
            pid_batch * stride_input_batch + 
            seq_idx * stride_input_seq + 
            hidden_offsets * stride_input_hidden
        )
        # Load values (masked)
        vals = tl.load(input_ptr + input_offsets, mask=mask_hidden, other=0.0)
        seq_sum += vals
    
    # Divide by seq_len to get mean
    result = seq_sum / seq_len
    
    # Store result
    output_offsets = (
        pid_batch * stride_output_batch + 
        hidden_offsets * stride_output_hidden
    )
    tl.store(output_ptr + output_offsets, result, mask=mask_hidden)


@torch.fx.wrap
def triton_mean_kernel_wrapper(in_3: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton mean kernel.
    """
    batch_size = in_3.shape[0]
    seq_len = in_3.shape[1]  # 49
    hidden_size = in_3.shape[2]  # 448
    
    # Allocate output
    output = torch.empty((batch_size, hidden_size), device=in_3.device, dtype=in_3.dtype)
    
    # Calculate grid
    # Grid: (batch_size, num_hidden_blocks)
    # Let autotuner pick the best BLOCK_SIZE
    grid = (batch_size, triton.cdiv(hidden_size, 64))
    
    # Launch kernel - BLOCK_SIZE handled by autotune
    triton_mean_kernel[grid](
        in_3, output,
        batch_size, seq_len, hidden_size,
        in_3.stride(0), in_3.stride(1), in_3.stride(2),
        output.stride(0), output.stride(1),
    )
    
    return output


def replacement_func():
    return triton_mean_kernel_wrapper