import torch
import triton
import triton.language as tl

# Pattern matching for Conv1D + Slice + GELU + Transpose
def pattern(x, weight, bias):
    # Conv1D operation
    conv1d = torch.conv1d(x, weight, bias, (1,), (64,), (1,), 16)
    
    # Slice operation (remove last element in sequence dimension)
    sliced = conv1d[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))]
    
    # GELU activation
    activated = torch.nn.functional.gelu(sliced)
    
    # Transpose operation
    transposed = activated.transpose(1, 2)
    
    # Return the transposed result that's used in the computation
    return transposed

# Arguments needed for the replacement
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Simplified fused kernel
@triton.jit
def fused_conv1d_gelu_transpose_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out_ptr, seq_idx_ptr,
    batch_size, in_channels, seq_len, out_channels,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate program IDs
    pid = tl.program_id(0)
    
    # Each program handles a specific element in the output
    batch_idx = pid // (out_channels * (seq_len - 1))
    channel_idx = (pid // (seq_len - 1)) % out_channels
    seq_idx = pid % (seq_len - 1)
    
    # Simple convolution for one output element
    conv_val = 0.0
    for c_in in range(0, in_channels, BLOCK_SIZE):
        for s in range(0, 128, BLOCK_SIZE):  # kernel_size is 128
            # Load input element
            x_offset = batch_idx * in_channels * seq_len + (c_in + (pid % BLOCK_SIZE)) * seq_len + seq_idx + s
            if (c_in + (pid % BLOCK_SIZE)) < in_channels and s < 128:
                x = tl.load(x_ptr + x_offset)
                weight_offset = channel_idx * in_channels * 128 + (c_in + (pid % BLOCK_SIZE)) * 128 + s
                w = tl.load(weight_ptr + weight_offset)
                conv_val += x * w
    
    # Apply bias
    bias_offset = channel_idx
    b = tl.load(bias_ptr + bias_offset)
    result = conv_val + b
    
    # GELU activation
    gelu_val = result * 0.5 * (1.0 + tl.sin(tl.constexpr(0.7978845608028654) * (result + 0.044715 * result * result * result)))
    
    # Store result directly in transposed format [batch, seq-1, out_channels]
    out_offset = batch_idx * (seq_len - 1) * out_channels + seq_idx * out_channels + channel_idx
    tl.store(out_ptr + out_offset, gelu_val)

@torch.fx.wrap
def fused_conv1d_gelu_transpose(x, weight, bias):
    batch_size, in_channels, seq_len = x.shape
    out_channels, _, _ = weight.shape
    
    # Calculate output dimensions (after slicing and transpose)
    output_seq_len = seq_len - 1
    
    # Allocate output tensor with final transpose shape [batch, seq-1, out_channels]
    output = torch.empty((batch_size, output_seq_len, out_channels), dtype=x.dtype, device=x.device)
    
    # Determine grid size (total elements in output)
    total_elements = batch_size * out_channels * output_seq_len
    BLOCK_SIZE = 32  # Use moderate block size for good utilization
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv1d_gelu_transpose_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        seq_idx_ptr=None,  # Not used in this simplified version
        batch_size=batch_size,
        in_channels=in_channels,
        seq_len=seq_len,
        out_channels=out_channels,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_conv1d_gelu_transpose