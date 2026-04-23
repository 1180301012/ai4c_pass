import torch
import triton
import triton.language as tl

# Pattern matching for softmax followed by redundant type conversion and dropout
@torch.fx.wrap
def pattern(matmul_result):
    softmax_out = torch.nn.functional.softmax(matmul_result, dim=-1, dtype=torch.float32)
    redundant_to_float = softmax_out.to(torch.float32)
    dropout_out = torch.nn.functional.dropout(redundant_to_float, p=0.0, training=False)
    return dropout_out

# Extract matmul_result as argument
@torch.fx.wrap
def replacement_args(matmul_result):
    return (matmul_result,)

# Triton softmax kernel optimized for 257-element sequences
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    stride_batch,
    stride_head,
    stride_seq,
    n_seq,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate grid parameters
    batch = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2)
    
    # Calculate row pointer
    row_start = batch * stride_batch + head * stride_head + seq * stride_seq
    input_row = input_ptr + row_start
    output_row = output_ptr + row_start
    
    # Load input values for the sequence (257 elements)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_seq
    x = tl.load(input_row + offsets, mask=mask, other=0.0)
    
    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    
    # Compute exponentials and sum
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    
    # Normalize and store result
    y = exp_x / sum_exp_x
    tl.store(output_row + offsets, y, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def softmax_wrapper(matmul_result):
    # Input shape: [batch=1, head=16, seq=257, 257]
    batch, head, seq, _ = matmul_result.shape
    n_seq = seq  # Last dimension for softmax
    
    # Allocate output tensor (float32)
    out = torch.empty_like(matmul_result, dtype=torch.float32)
    
    # Set up grid dimensions: (batch, head, seq)
    grid = (batch, head, seq)
    
    # Launch kernel (BLOCK_SIZE = 256 for 257-element sequences)
    softmax_kernel[grid](
        matmul_result,
        out,
        matmul_result.stride(0),
        matmul_result.stride(1),
        matmul_result.stride(2),
        n_seq,
        BLOCK_SIZE=256
    )
    
    return out

# Return the kernel wrapper as the replacement
@torch.fx.wrap
def replacement_func():
    return softmax_wrapper