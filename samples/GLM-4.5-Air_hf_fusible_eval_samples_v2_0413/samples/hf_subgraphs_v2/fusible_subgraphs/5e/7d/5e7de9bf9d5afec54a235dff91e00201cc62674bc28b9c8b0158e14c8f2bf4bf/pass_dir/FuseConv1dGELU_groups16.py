import torch
import triton
import triton.language as tl

# Pattern matching function - Conv1d + GELU fusion
def pattern(x, weight, bias, stride, padding, dilation, groups):
    # Conv1d operation - using positional arguments exactly as in original
    conv = torch.conv1d(x, weight, bias, stride, padding, dilation, groups)
    # GELU activation
    gelu_out = torch.nn.functional.gelu(conv)
    return gelu_out

# Argument extraction function
def replacement_args(x, weight, bias, stride, padding, dilation, groups):
    return (x, weight, bias, stride, padding, dilation, groups)

# Debug kernel - Just zero the output to test size
@triton.jit
def conv1d_gelu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C_out, L_out, C_in, kernel_size
):
    pid = tl.program_id(0)
    
    # Calculate which output element we're processing
    n = pid // (C_out * L_out)
    c = (pid % (C_out * L_out)) // L_out  
    l = pid % L_out
    
    # Skip if out of bounds
    if n >= N or c >= C_out or l >= L_out:
        return
    
    # Just store zero to test output size
    output_ptr = n * C_out * L_out + c * L_out + l
    tl.store(out_ptr + output_ptr, 0.0)

# Kernel wrapper
@torch.fx.wrap
def conv1d_gelu_fused(x, weight, bias, stride, padding, dilation, groups):
    # Get tensor shapes
    batch_size, in_channels, in_length = x.shape
    out_channels, kernel_size, _ = weight.shape
    
    # Use the exact standard conv1d output length formula
    stride_val = stride[0] if isinstance(stride, tuple) else stride
    padding_val = padding[0] if isinstance(padding, tuple) else padding
    dilation_val = dilation[0] if isinstance(dilation, tuple) else dilation
    out_length = (in_length + 2 * padding_val - dilation_val * (kernel_size - 1)) // stride_val + 1
    
    # Total number of elements in output
    total_elements = batch_size * out_channels * out_length
    
    # Grid configuration - using simple 1D grid with block size 1024
    num_programs = (total_elements + 1023) // 1024
    
    # Output tensor
    out = torch.empty((batch_size, out_channels, out_length), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    conv1d_gelu_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=batch_size,
        C_out=out_channels,
        L_out=out_length,
        C_in=in_channels,
        kernel_size=kernel_size
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return conv1d_gelu_fused