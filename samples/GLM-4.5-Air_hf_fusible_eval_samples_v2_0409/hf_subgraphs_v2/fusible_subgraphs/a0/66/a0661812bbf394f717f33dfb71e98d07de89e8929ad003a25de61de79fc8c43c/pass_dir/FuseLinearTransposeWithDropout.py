import torch
import triton
import triton.language as tl

# Pattern matching function - exact match to target pattern
def pattern(in_0, in_1, in_2):
    """Match exact linear + dropout + transpose pattern"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.1, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_3, tmp_4)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# High-performance fused kernel
@triton.jit
def fused_kernel_with_dropout(
    bias_ptr,
    weight_ptr,
    input_ptr,
    out_ptr,
    transpose_out_ptr,
    dropout_p,
    batch_size,
    seq_len,
    input_features,
    output_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Calculate program IDs
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Calculate memory offsets
    input_offset = batch_id * seq_len * input_features + seq_id * input_features
    out_offset = batch_id * seq_len * output_features + seq_id * output_features
    transpose_offset = batch_id * output_features * seq_len + seq_id
    
    # Load bias
    bias = tl.load(bias_ptr + tl.arange(0, output_features))
    
    # Process input features in blocks
    for k in range(0, input_features, BLOCK_SIZE_K):
        # Load input and weight blocks
        input_start = input_offset + k
        weight_start = k * output_features
        
        # Bounds checking for loading
        input_mask = (k + tl.arange(0, BLOCK_SIZE_K)) < input_features
        weight_mask = (k + tl.arange(0, BLOCK_SIZE_K)) < input_features
        
        # Load data
        input_data = tl.load(input_ptr + input_start + tl.arange(0, BLOCK_SIZE_K), 
                           mask=input_mask, other=0.0)
        weights = tl.load(weight_ptr + weight_start + tl.arange(0, BLOCK_SIZE_K) * output_features,
                        mask=weight_mask, other=0.0).to(tl.float32)
        
        # Matrix multiplication and bias
        acc = tl.dot(input_data, weights) + bias
        
        # Apply dropout if needed
        if dropout_p > 0:
            # Generate dropout mask using triton random number generation
            dropout_mask = tl.rand(output_features) > dropout_p
            acc = acc * dropout_mask
        
        # Store results
        store_mask = tl.arange(0, output_features) < output_features
        tl.store(out_ptr + out_offset + tl.arange(0, output_features),
                acc, mask=store_mask)
        tl.store(transpose_out_ptr + transpose_offset + tl.arange(0, output_features) * seq_len,
                acc, mask=store_mask)

# Kernel wrapper
@torch.fx.wrap
def fused_linear_transpose_with_dropout(in_0, in_1, in_2):
    batch_size, seq_len, input_features = in_2.shape
    output_features = in_0.shape[0]
    
    # Output tensors
    linear_out = torch.empty(batch_size, seq_len, output_features, dtype=in_2.dtype, device=in_2.device)
    transpose_out = torch.empty(batch_size, output_features, seq_len, dtype=in_2.dtype, device=in_2.device)
    
    # Fixed dropout rate matches the pattern
    dropout_p = 0.1
    
    # Optimized block sizes
    if dropout_p > 0:
        BLOCK_SIZE_K = 32  # Smaller block size for dropout to avoid excessive random number generation
    else:
        BLOCK_SIZE_K = 64  # Larger block size for no-dropout case
    
    BLOCK_SIZE_M = 1  # Process one sequence element per program
    BLOCK_SIZE_N = min(128, output_features)  # Adjust based on output features
    
    # Calculate grid dimensions
    grid = (batch_size, seq_len)
    
    # Launch kernel
    fused_kernel_with_dropout[grid](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        out_ptr=linear_out,
        transpose_out_ptr=transpose_out,
        dropout_p=dropout_p,
        batch_size=batch_size,
        seq_len=seq_len,
        input_features=input_features,
        output_features=output_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return linear_out, transpose_out

# Replacement function
def replacement_func():
    return fused_linear_transpose_with_dropout