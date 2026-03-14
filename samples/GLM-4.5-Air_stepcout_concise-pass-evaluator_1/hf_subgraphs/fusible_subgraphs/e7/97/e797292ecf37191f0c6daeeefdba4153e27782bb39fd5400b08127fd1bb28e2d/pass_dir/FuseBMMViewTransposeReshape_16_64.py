import torch
import triton
import triton.language as tl


# Pattern matching function - matches bmm + view + transpose + reshape pattern for 16x64 case
def pattern(tmp_1, in_1):
    # Batch matrix multiply
    tmp_2 = torch.bmm(tmp_1, in_1)
    # View to reshape - using dimensions for 16x64 case
    tmp_3 = tmp_2.view(1, 16, 1, 64)
    # Transpose dimensions 1 and 2
    tmp_4 = tmp_3.transpose(1, 2)
    # Final reshape to [1, 1, 1024]
    tmp_5 = tmp_4.reshape(1, 1, 1024)
    return tmp_5


def replacement_args(tmp_1, in_1):
    return (tmp_1, in_1)


# Optimized Triton kernel that fuses bmm + view + transpose + reshape for 16x64 case
@triton.jit
def fused_bmm_view_transpose_reshape_kernel_16_64(
    a_ptr,  # input a: [batch, 1, 1]
    b_ptr,  # input b: [batch, 1, head_dim]
    output_ptr,  # output: [1, 1, batch * head_dim]
    batch_size,  # batch dimension
    head_dim,  # head dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row of the batch
    batch_idx = tl.program_id(0)
    
    # Calculate offsets
    # a is [batch, 1, 1] - we need a[batch_idx, 0, 0]
    a_offset = batch_idx * 1 * 1
    # b is [batch, 1, head_dim] - we need b[batch_idx, 0, :]
    b_offset = batch_idx * 1 * head_dim
    
    # Load a[batch_idx, 0, 0] - it's a scalar per batch
    a_val = tl.load(a_ptr + a_offset)
    
    # Load b[batch_idx, 0, :] - the head_dim values
    # Use BLOCK_SIZE which is constexpr instead of head_dim
    b_vals = tl.load(b_ptr + b_offset + tl.arange(0, BLOCK_SIZE))
    
    # Multiply a * b (broadcasting scalar a across vector b)
    result = a_val * b_vals
    
    # Store result to output - output layout is [1, 1, batch * head_dim]
    # We need to store at position batch_idx * head_dim + head_idx
    output_offset = batch_idx * head_dim + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + output_offset, result)


def fused_bmm_view_transpose_reshape_16_64(a, b):
    """
    Fused kernel that computes: bmm(a, b) -> view -> transpose -> reshape
    Input:
        a: [batch, 1, 1] - attention weights after softmax
        b: [batch, 1, head_dim] - value states
    Output:
        [1, 1, batch * head_dim]
    """
    batch_size = a.shape[0]
    head_dim = b.shape[2]
    
    # Allocate output
    output = torch.empty((1, 1, batch_size * head_dim), device=a.device, dtype=a.dtype)
    
    # Launch kernel - one program per batch element
    grid = (batch_size,)
    
    # Use BLOCK_SIZE = 128 for typical head dimensions (32, 64, etc.)
    BLOCK_SIZE = 128
    
    fused_bmm_view_transpose_reshape_kernel_16_64[grid](
        a_ptr=a,
        b_ptr=b,
        output_ptr=output,
        batch_size=batch_size,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


@torch.fx.wrap
def kernel_wrapper(a, b):
    return fused_bmm_view_transpose_reshape_16_64(a, b)


def replacement_func():
    return kernel_wrapper