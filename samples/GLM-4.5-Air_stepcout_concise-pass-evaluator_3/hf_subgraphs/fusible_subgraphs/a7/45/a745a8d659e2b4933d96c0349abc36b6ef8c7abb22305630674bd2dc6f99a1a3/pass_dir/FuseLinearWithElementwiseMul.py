import torch
import triton
import triton.language as tl

# Pattern matching function for transformers case: Linear + Element-wise Multiplication  
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    # Match the computation structure with simple operations
    tmp_1 = in_1  # Will be matched by the linear operation in the graph
    tmp_0 = None
    tmp_2 = in_2 * tmp_1
    tmp_1 = None
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel for linear + element-wise multiplication fusion
@triton.jit
def fused_linear_mul_kernel(
    weight_ptr,           # [out_features, in_features]
    input_1_ptr,          # [batch_size, seq_len, in_features] 
    input_2_ptr,          # [batch_size, seq_len, out_features]
    output_ptr,           # [batch_size, seq_len, out_features]
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
    out_ptr = output_ptr + base_idx_out
    
    # Compute input addresses
    base_idx_in = batch_idx * seq_len * in_features + seq_idx * in_features
    in_ptr = input_1_ptr + base_idx_in
    
    # Load input features
    in_ptrs = in_ptr + tl.arange(0, in_features)
    inputs = tl.load(in_ptrs, mask=tl.arange(0, in_features) < in_features, other=0.0)
    
    # Load multiplication factor
    mul_ptrs = input_2_ptr + base_idx_out + feature_idx
    mul_factors = tl.load(mul_ptrs, mask=feature_idx < out_features, other=0.0)
    
    # Compute linear transformation using matrix multiplication
    linear_output = tl.zeros([BLOCK_SIZE_BATCH], dtype=tl.float32)
    
    if in_features <= 1024 and BLOCK_SIZE_BATCH <= 64:  # Medium to small workload
        # Use loop for better utilization
        for k in range(0, in_features, 128):
            end_k = min(k + 128, in_features)
            in_k = tl.load(in_ptrs + k, mask=tl.arange(k, end_k) < in_features, other=0.0)
            
            # Load corresponding weight section
            weight_ptrs_k = weight_ptr + (feature_idx[:, None] * in_features + tl.arange(k, end_k)[None, :])
            weight_k = tl.load(weight_ptrs_k, 
                              mask=(feature_idx[:, None] < out_features) & (tl.arange(k, end_k)[None, :] < in_features), 
                              other=0.0)
            
            # Accumulate dot product
            linear_output += tl.dot(in_k.to(tl.float32), weight_k.to(tl.float32)).to(tl.float32)
    else:  # Large workload vectorized
        # Vectorized matrix multiplication
        weight_transposed = weight_ptr + (tl.arange(0, BLOCK_SIZE_BATCH).to(tl.pointer_type())[:, None] * in_features + 
                                          tl.arange(0, in_features)[None, :])
        weight = tl.load(weight_transposed, 
                        mask=(tl.arange(0, BLOCK_SIZE_BATCH)[:, None] < out_features) & 
                              (tl.arange(0, in_features)[None, :] < in_features), 
                        other=0.0)
        
        linear_output = tl.sum(inputs.to(tl.float32) * weight.to(tl.float32), axis=1)
    
    # Apply element-wise multiplication
    final_output = linear_output * mul_factors.to(tl.float32)
    
    # Store final result
    tl.store(out_ptr + feature_idx, final_output.to(tl.float32), mask=feature_idx < out_features)

@torch.fx.wrap
def fused_linear_mul_module(in_0, in_1, in_2):
    # Get tensor shapes and determine launch configuration
    batch_size, seq_len, in_features = in_1.shape
    out_features = in_0.shape[0]
    
    # Create output tensor
    output = torch.empty_like(in_1, dtype=torch.float32)  # Same shape as linear output
    
    # Launch Triton kernel
    BLOCK_SIZE_BATCH = 128  # Optimal for feature-wise parallelism
    BLOCK_SIZE_SEQ = 256    # Optimal for sequence-wise parallelism
    
    # Grid calculation for 3D grid (batch, seq, features)
    grid = (
        (batch_size + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ,
        (seq_len + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ,
        (out_features + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH,
    )
    
    fused_linear_mul_kernel[grid](
        in_0,
        in_1,
        in_2, 
        output,
        batch_size, seq_len, out_features, in_features,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
    )
    
    return (output,)

# Function for bfloat16 support (for transformers model)
@triton.jit
def fused_linear_mul_kernel_bf16(
    weight_ptr,
    input_1_ptr,
    input_2_ptr, 
    output_ptr,
    batch_size, seq_len, out_features, in_features,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    # Same kernel structure but optimized for bfloat16
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    feature_idx = tl.program_id(2) * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    
    base_idx_out = batch_idx * seq_len * out_features + seq_idx * out_features
    base_idx_in = batch_idx * seq_len * in_features + seq_idx * in_features
    
    in_ptr = input_1_ptr + base_idx_in
    out_ptr = output_ptr + base_idx_out
    
    # Load inputs 
    in_ptrs = in_ptr + tl.arange(0, in_features)
    inputs = tl.load(in_ptrs, mask=tl.arange(0, in_features) < in_features, other=0.0)
    
    mul_ptrs = input_2_ptr + base_idx_out + feature_idx
    mul_factors = tl.load(mul_ptrs, mask=feature_idx < out_features, other=0.0)
    
    # Compute linear transformation with bfloat16 support
    linear_output = tl.zeros([BLOCK_SIZE_BATCH], dtype=tl.bfloat16)
    
    # Optimized for medium-sized dimensions (typical in transformers)
    weight_ptrs = weight_ptr + (tl.arange(0, BLOCK_SIZE_BATCH).to(tl.pointer_type())[:, None] * in_features + 
                                tl.arange(0, in_features)[None, :])
    weight = tl.load(weight_ptrs, 
                    mask=(tl.arange(0, BLOCK_SIZE_BATCH)[:, None] < out_features) & 
                          (tl.arange(0, in_features)[None, :] < in_features), 
                    other=0.0)
    
    # Fast matrix multiplication
    linear_output = tl.dot(inputs.to(tl.bfloat16), weight.to(tl.bfloat16))
    
    # Apply element-wise multiplication
    final_output = linear_output * mul_factors.to(tl.bfloat16)
    
    # Store result
    tl.store(out_ptr + feature_idx, final_output, mask=feature_idx < out_features)

@torch.fx.wrap  
def fused_linear_mul_module_bf16(in_0, in_1, in_2):
    # Get tensor shapes
    batch_size, seq_len, in_features = in_1.shape
    out_features = in_0.shape[0]
    
    # Create output tensor
    output = torch.empty(batch_size, seq_len, out_features, dtype=torch.bfloat16, device=in_1.device)
    
    # Launch kernel - use larger blocks for bfloat16 precision
    BLOCK_SIZE_BATCH = 256  # Larger blocks for better occupancy with bfloat16
    BLOCK_SIZE_SEQ = 128
    
    grid = (
        (batch_size + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ,
        (seq_len + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ, 
        (out_features + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH,
    )
    
    fused_linear_mul_kernel_bf16[grid](
        in_0,
        in_1,
        in_2,
        output,
        batch_size, seq_len, out_features, in_features,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
    )
    
    return (output,)

# Kernel selector - chooses appropriate kernel based on data type
@torch.fx.wrap
def fused_linear_mul_selector(in_0, in_1, in_2):
    if in_1.dtype == torch.bfloat16:
        return fused_linear_mul_module_bf16(in_0, in_1, in_2)
    else:
        return fused_linear_mul_module(in_0, in_1, in_2)

# Replacement function (MUST be zero-argument, return function reference)
def replacement_func():
    return fused_linear_mul_selector