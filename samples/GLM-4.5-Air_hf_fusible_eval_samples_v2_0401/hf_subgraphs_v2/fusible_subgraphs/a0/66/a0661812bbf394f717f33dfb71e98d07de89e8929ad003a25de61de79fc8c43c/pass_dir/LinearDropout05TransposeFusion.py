import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern for linear -> dropout(p=0.05) -> transpose"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    dropout = torch.nn.functional.dropout(linear, 0.05, False, False)
    transpose = dropout.transpose(1, 2)
    return dropout, transpose

@triton.jit
def linear_dropout05_transpose_kernel(
    x_ptr,           # in_2: [batch, seq_len, input_features]
    weight_ptr,      # in_1: [output_features, input_features] 
    bias_ptr,        # in_0: [output_features]
    out_ptr1,        # dropout output: [batch, seq_len, output_features]
    out_ptr2,        # transposed output: [batch, output_features, seq_len]
    batch_size,
    seq_len, 
    input_features,
    output_features,
):
    """Fused linear + dropout(0.05) + transpose kernel - simplified version"""
    pid = tl.program_id(0)
    
    # Simple 1D grid - each program handles one output feature
    if pid >= output_features:
        return
        
    # Initialize accumulator for this output feature
    acc = tl.zeros([seq_len], dtype=tl.float32)
    
    # Compute linear transformation for this output feature
    for k in range(input_features):
        # Load weight element
        weight_val = tl.load(weight_ptr + pid * input_features + k, mask=(k < input_features), other=0.0)
        
        # Load input values for all sequence positions
        input_ptr = x_ptr + k
        input_vals = tl.load(input_ptr + tl.arange(0, seq_len) * input_features, mask=(k < input_features), other=0.0)
        
        # Accumulate dot product
        acc += input_vals * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + pid)
    acc = acc + bias_val
    
    # Apply dropout with p=0.05 (inference mode: scale by 1/0.95)
    dropout_p = 0.05
    scale = 1.0 / (1.0 - dropout_p)
    # Generate random mask for dropout (deterministic in inference)
    seed = pid  # Simple seed for reproducibility
    rand_vals = tl.rand(tl.arange(0, seq_len), seed=seed)
    dropout_mask = rand_vals > dropout_p
    dropout_mask = dropout_mask.to(tl.float32)
    acc = acc * dropout_mask * scale
    
    # Cast to float16 for output
    result = acc.to(tl.float16)
    
    # Store outputs
    # Output 1: [batch, seq_len, output_features]
    seq_offsets = tl.arange(0, seq_len)
    out1_offsets = (seq_offsets * output_features + pid)
    tl.store(out_ptr1 + out1_offsets, result)
    
    # Output 2: [batch, output_features, seq_len]
    out2_offsets = (pid * seq_len + seq_offsets)
    tl.store(out_ptr2 + out2_offsets, result)

@torch.fx.wrap
def linear_dropout05_transpose(in_0, in_1, in_2):
    """Wrapper for fused linear + dropout(0.05) + transpose"""
    batch_size, seq_len, input_features = in_2.shape
    output_features = in_0.shape[0]
    
    # Output shapes
    out1_shape = (batch_size, seq_len, output_features)
    out2_shape = (batch_size, output_features, seq_len)
    
    # Allocate outputs with correct dtype
    out1 = torch.empty(out1_shape, dtype=in_2.dtype, device=in_2.device)
    out2 = torch.empty(out2_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Launch simplified kernel - one program per output feature
    grid = output_features
    
    linear_dropout05_transpose_kernel[grid](
        in_2,
        in_1,
        in_0,
        out1,
        out2,
        batch_size,
        seq_len,
        input_features,
        output_features
    )
    
    return out1, out2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return linear_dropout05_transpose