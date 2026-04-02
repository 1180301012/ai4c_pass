import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Debug pattern: linear only (without dropout)"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear

@triton.jit
def linear_kernel_debug(
    x_ptr,           # in_2: [batch, seq_len, input_features]
    weight_ptr,      # in_1: [output_features, input_features] 
    bias_ptr,        # in_0: [output_features]
    out_ptr,         # output: [batch, seq_len, output_features]
    batch_size,
    seq_len, 
    input_features,
    output_features,
    MAX_SEQ_LEN: tl.constexpr,
):
    """Simple linear kernel for debugging"""
    pid = tl.program_id(0)
    
    # Simple 1D grid - each program handles one output feature
    if pid >= output_features:
        return
        
    # Initialize accumulator for this output feature (with max size)
    acc = tl.zeros([MAX_SEQ_LEN], dtype=tl.float32)
    
    # Compute linear transformation for this output feature
    for k in range(input_features):
        # Load weight element
        weight_val = tl.load(weight_ptr + pid * input_features + k, mask=(k < input_features), other=0.0)
        
        # Load input values for all sequence positions with proper masking
        input_ptr = x_ptr + k
        input_vals = tl.load(input_ptr + tl.arange(0, MAX_SEQ_LEN) * input_features, 
                           mask=(tl.arange(0, MAX_SEQ_LEN) < seq_len) & (k < input_features), 
                           other=0.0)
        
        # Accumulate dot product
        acc += input_vals * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + pid, mask=(pid < output_features), other=0.0)
    acc = acc + bias_val
    
    # Apply masking to only store valid sequence positions
    mask = tl.arange(0, MAX_SEQ_LEN) < seq_len
    
    # Cast to float16 for output and apply mask
    result = acc.to(tl.float16)
    
    # Store output with proper masking
    seq_offsets = tl.arange(0, MAX_SEQ_LEN)
    out_offsets = (seq_offsets * output_features + pid)
    tl.store(out_ptr + out_offsets, result, mask=mask)

@torch.fx.wrap
def linear_debug(in_0, in_1, in_2):
    """Wrapper for simple linear transformation"""
    batch_size, seq_len, input_features = in_2.shape
    output_features = in_0.shape[0]
    
    # Output shape
    out_shape = (batch_size, seq_len, output_features)
    
    # Allocate output
    out = torch.empty(out_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Determine maximum sequence length for compile-time constant
    # Using a power of 2 that can accommodate typical sequence lengths
    MAX_SEQ_LEN = 2048  # Power of 2 that can handle seq_len up to 2048
    
    # Launch kernel
    grid = (output_features,)  # Wrap in tuple for 1D grid
    linear_kernel_debug[grid](
        in_2,
        in_1,
        in_0,
        out,
        batch_size,
        seq_len,
        input_features,
        output_features,
        MAX_SEQ_LEN
    )
    
    return out

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return linear_debug