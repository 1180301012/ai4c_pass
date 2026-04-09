import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """Match linear + dropout + transpose pattern"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.1, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_3, tmp_4)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# High-performance fused kernel
@triton.jit
def fused_linear_dropout_transpose_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    out_ptr,
    transpose_out_ptr,
    batch_size,
    seq_len,
    input_features,
    output_features,
    dropout_p: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Calculate program IDs for batch and sequence dimensions
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Calculate memory pointers for this program
    bias_offset = batch_id * seq_len * output_features
    input_offset = batch_id * seq_len * input_features
    out_offset = batch_id * seq_len * output_features
    transpose_offset = batch_id * output_features * seq_len
    
    # Shared memory for bias (broadcast across batch and sequence)
    bias_shared = tl.load(bias_ptr + tl.arange(0, output_features))
    bias_shared = tl.broadcast_to(bias_shared, [seq_len, output_features])
    
    # Process sequence elements in blocks
    for k in range(0, input_features, BLOCK_SIZE_K):
        # Load input block and weights
        input_start = input_offset + seq_id * input_features + k
        weight_start = k * output_features
        
        # Load input data with bounds checking
        input_mask = (k + tl.arange(0, BLOCK_SIZE_K)) < input_features
        input_data = tl.load(input_ptr + input_start + tl.arange(0, BLOCK_SIZE_K), 
                           mask=input_mask, other=0.0)
        
        # Load weights with bounds checking
        weight_mask = (k + tl.arange(0, BLOCK_SIZE_K)) < input_features
        weights = tl.load(weight_ptr + weight_start + tl.arange(0, BLOCK_SIZE_K) * output_features,
                        mask=weight_mask, other=0.0).to(tl.float32)
        
        # Matrix multiplication: output = input * weight^T
        acc = tl.dot(input_data, weights)
        
        # Add bias
        acc = acc + bias_shared[seq_id, tl.arange(0, output_features)]
        
        # Apply dropout
        if dropout_p > 0:
            dropout_mask = tl.rand(output_features) > dropout_p
            acc = acc * dropout_mask
        
        # Store results
        store_mask = tl.arange(0, output_features) < output_features
        tl.store(out_ptr + out_offset + seq_id * output_features + tl.arange(0, output_features),
                acc, mask=store_mask)
        
        # Store transpose result directly
        tl.store(transpose_out_ptr + transpose_offset + tl.arange(0, output_features) * seq_len + seq_id,
                acc, mask=store_mask)

# Kernel wrapper
@torch.fx.wrap  
def fused_linear_dropout_transpose(in_0, in_1, in_2, dropout_p=0.1):
    batch_size, seq_len, input_features = in_2.shape
    output_features = in_0.shape[0]
    
    # Output shapes
    linear_out = torch.empty(batch_size, seq_len, output_features, dtype=in_2.dtype, device=in_2.device)
    transpose_out = torch.empty(batch_size, output_features, seq_len, dtype=in_2.dtype, device=in_2.device)
    
    # Block sizes for better GPU utilization
    BLOCK_SIZE_M = 1  # Process one sequence element at a time for simplicity
    BLOCK_SIZE_N = 128  # Output features block size
    BLOCK_SIZE_K = 32   # Input features block size
    
    # Calculate grid dimensions
    grid = (batch_size, seq_len)
    
    # Launch the fused kernel
    fused_linear_dropout_transpose_kernel[grid](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        out_ptr=linear_out,
        transpose_out_ptr=transpose_out,
        batch_size=batch_size,
        seq_len=seq_len,
        input_features=input_features,
        output_features=output_features,
        dropout_p=dropout_p,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return linear_out, transpose_out

# Replacement function
def replacement_func():
    return fused_linear_dropout_transpose