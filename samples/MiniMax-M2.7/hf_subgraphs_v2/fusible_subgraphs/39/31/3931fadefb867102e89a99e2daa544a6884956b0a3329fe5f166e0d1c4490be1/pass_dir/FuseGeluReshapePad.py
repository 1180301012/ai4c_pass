import torch
import triton
import triton.language as tl


# Constants for GELU approximation
SQRT_2_OVER_PI = 0.7978845608
GELU_C = 0.044715


@triton.jit
def gelu(x):
    """GELU activation using tanh approximation."""
    x_cubed = x * x * x
    inner = SQRT_2_OVER_PI * (x + GELU_C * x_cubed)
    return 0.5 * x * (1.0 + tl.math.tanh(inner))


@triton.jit
def gelu_reshape_pad_kernel(
    input_ptr,
    output_ptr,
    output_batch_stride: tl.constexpr,
    output_seq_stride: tl.constexpr,
    output_dim_stride: tl.constexpr,
    input_seq_stride: tl.constexpr,
    input_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GELU + Reshape + Pad kernel
    
    Input shape: [batch, seq, dim] = [1, 124, 1536]
    Output shape: [batch, 249, dim] = [1, 249, 768]
    
    The reshape transforms [1, 124, 1536] -> [1, 124, 2, 768] -> [1, 248, 768]
    Then pad adds 1 row at bottom: [1, 248, 768] -> [1, 249, 768]
    
    Index mapping for output[b, s, d]:
    - If s < 248: corresponds to input[b, s // 2, (s % 2) * 768 + d]
    - If s == 248: zero (padded)
    """
    pid = tl.program_id(0)
    
    # Each program handles a contiguous block of output elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (249 * 768)  # Total output elements
    
    # Calculate output position from linear offset
    # output[batch, seq, dim] where batch=0, dim=768
    seq_idx = offsets // 768
    dim_idx = offsets % 768
    
    # Check if this is a padded position (seq_idx == 248)
    is_padded = seq_idx == 248
    
    # For non-padded positions, compute input index
    # input[b, s // 2, (s % 2) * 768 + d]
    input_seq_idx = seq_idx // 2
    input_inner_idx = (seq_idx % 2) * 768 + dim_idx
    
    # Compute linear input index
    input_offset = input_seq_idx * input_dim + input_inner_idx
    
    # Load input value (zeros for padded positions handled by is_padded)
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Apply GELU
    result = gelu(x)
    
    # Zero out padded positions
    result = tl.where(is_padded, 0.0, result)
    
    # Store to output
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def gelu_reshape_pad_wrapper(x):
    """
    Wrapper function for the fused GELU + Reshape + Pad operation.
    
    Input: [1, 124, 1536] - bfloat16/float16
    Output: [1, 249, 768] - bfloat16/float16
    
    The transformation:
    1. Reshape input to [1, 124, 2, 768]
    2. Reshape to [1, 248, 768]
    3. Apply GELU
    4. Pad with zeros to get [1, 249, 768]
    """
    batch, seq, dim = x.shape  # [1, 124, 1536]
    
    # Output shape after pad: [1, 249, 768]
    output_seq = 249
    output_dim = 768
    total_output_elements = batch * output_seq * output_dim
    
    # Block size for Triton
    BLOCK_SIZE = 1024
    num_programs = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty((batch, output_seq, output_dim), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    gelu_reshape_pad_kernel[(num_programs,)](
        x,
        out,
        output_batch_stride=output_seq * output_dim,
        output_seq_stride=output_dim,
        output_dim_stride=1,
        input_seq_stride=dim,
        input_dim=dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0):
    """
    Match the pattern: gelu -> reshape -> reshape -> pad
    """
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return gelu_reshape_pad_wrapper