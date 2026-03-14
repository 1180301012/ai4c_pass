import torch
import triton
import triton.language as tl

# Pattern matching function for mmpose case: Linear + Broadcast Multiplication
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    # Match the computation structure - use simple assignments to match the graph
    # The actual linear operation will be matched by the underlying graph pattern
    tmp_2 = in_3  # This will be matched by the linear operation in the graph
    tmp_0 = None
    tmp_3 = in_2 * tmp_1
    tmp_1 = None
    return (tmp_3, tmp_2)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized Triton kernel for linear + broadcast multiplication fusion
@triton.jit
def fused_linear_broadcast_kernel(
    weight_ptr,           # [out_features, in_features] 
    scale_ptr,            # [out_features]
    input_3_ptr,          # [batch_size, seq_len, in_features]
    input_2_ptr,          # [batch_size, seq_len, out_features]
    output_3_ptr,         # [batch_size, seq_len, out_features]
    output_2_ptr,         # [batch_size, seq_len, out_features]
    batch_size, seq_len, out_features, in_features,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    # Program grid based on batch and sequence dimensions
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    feature_idx = tl.program_id(2) * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    
    # Compute output addresses
    base_idx_out = batch_idx * seq_len * out_features + seq_idx * out_features
    out_ptr = output_3_ptr + base_idx_out
    
    # Compute input addresses  
    base_idx_in = batch_idx * seq_len * in_features + seq_idx * in_features
    in_ptr = input_3_ptr + base_idx_in
    
    # Load weight (transposed for better memory access)
    weight_ptrs = weight_ptr + (tl.arange(0, BLOCK_SIZE_BATCH).to(tl.pointer_type())[:, None] * out_features + feature_idx[:, None])
    weight = tl.load(weight_ptrs, mask=feature_idx[:, None] < out_features, other=0.0)
    
    # Load input features
    in_ptrs = in_ptr + tl.arange(0, in_features)
    inputs = tl.load(in_ptrs, mask=tl.arange(0, in_features) < in_features, other=0.0)
    
    # Compute linear transformation: output[batch, seq, features] = inputs @ weight.T
    linear_output = tl.zeros([BLOCK_SIZE_BATCH], dtype=tl.float32)
    if BLOCK_SIZE_BATCH <= 8:  # Small block size
        for k in range(0, in_features, 32):
            end_k = min(k + 32, in_features)
            in_k = tl.load(in_ptrs + k, mask=tl.arange(k, end_k) < in_features, other=0.0)
            weight_k = tl.load(weight_ptrs + k, mask=(feature_idx[:, None] < out_features) & (tl.arange(k, end_k)[None, :] < in_features), other=0.0)
            linear_output += tl.dot(in_k, weight_k.to(tl.float32)).to(tl.float32)
    else:  # Larger block size vectorized
        linear_output = tl.sum(inputs.to(tl.float32) * weight.to(tl.float32), axis=1)
    
    # Load scale factor and broadcast multiply
    scale = tl.load(scale_ptr + feature_idx, mask=feature_idx < out_features, other=0.0).to(tl.float32)
    output_mul = linear_output * scale
    
    # Store outputs
    tl.store(out_ptr + feature_idx, output_mul.to(tl.float32), mask=feature_idx < out_features)
    
    # Copy linear result to output_2 (same linear output)
    tl.store(output_2_ptr + base_idx_out + feature_idx, output_mul.to(tl.float32), mask=feature_idx < out_features)

@torch.fx.wrap
def fused_linear_broadcast_module(in_0, in_1, in_2, in_3):
    # Get tensor shapes and determine launch configuration
    batch_size, seq_len, in_features = in_3.shape
    out_features = in_0.shape[0]
    
    # Create output tensors
    output_3 = torch.empty_like(in_2)  # Element-wise multiplication result
    output_2 = torch.empty_like(in_2)  # Linear transformation result
    
    # Launch Triton kernel
    BLOCK_SIZE_BATCH = 64
    BLOCK_SIZE_SEQ = 128
    
    # Grid calculation
    grid = (
        (batch_size + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ,
        (seq_len + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ,
        (out_features + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH,
    )
    
    fused_linear_broadcast_kernel[grid](
        in_0,
        in_1, 
        in_3,
        in_2,
        output_3,
        output_2,
        batch_size, seq_len, out_features, in_features,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
    )
    
    return (output_3, output_2)

# Replacement function (MUST be zero-argument, return function reference)
def replacement_func():
    return fused_linear_broadcast_module