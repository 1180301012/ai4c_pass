import torch
import triton
import triton.language as tl

# Pattern matching function for mmpose: linear + element-wise multiplication
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_3, tmp_0, None)
    tmp_3 = in_2 * tmp_1
    return (tmp_3, tmp_2)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel for optimized linear + multiplication
@triton.jit
def linear_mul_kernel(
    weight_ptr,
    scale_ptr,
    input_a_ptr,
    input_b_ptr,
    output_mul_ptr,
    output_linear_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Compute program ID and offsets
    pid = tl.program_id(0)
    num_batches = batch_size * seq_len
    
    # Handle batch dimension
    batch_id = pid // num_batches
    seq_id = (pid % num_batches) // 1 if num_batches > 1 else 0
    
    # Output offsets
    mul_out_offset = batch_id * seq_len * in_features + seq_id * in_features
    linear_out_offset = batch_id * seq_len * out_features + seq_id * out_features
    
    # Load scale for broadcasting
    scale = tl.load(scale_ptr + batch_id * 256)
    
    # Process output_a: scaling with broadcast
    offsets = mul_out_offset + tl.arange(0, BLOCK_SIZE_M)
    mask = (offsets < (batch_id * seq_len + seq_id + 1) * in_features) if num_batches > 1 else (offsets < in_features)
    
    input_a = tl.load(input_a_ptr + offsets, mask=mask, other=0.0)
    output_mul = input_a * scale
    tl.store(output_mul_ptr + offsets, output_mul, mask=mask)
    
    # Process linear operation with matrix multiplication optimization
    # Simplified linear operation - for full optimization would need more complex kernel
    # Here we focus on optimizing the memory access patterns
    if out_features <= BLOCK_SIZE_N:
        # Small matrix optimization
        offsets_out = linear_out_offset + tl.arange(0, BLOCK_SIZE_N)
        mask_out = (offsets_out < (batch_id * seq_len + seq_id + 1) * out_features) if num_batches > 1 else (offsets_out < out_features)
        
        # Load input slice
        offsets_in = (batch_id * seq_len * in_features + seq_id * in_features) + tl.arange(0, BLOCK_SIZE_K)
        mask_in = (offsets_in < (batch_id * seq_len + seq_id + 1) * in_features) if num_batches > 1 else (offsets_in < in_features)
        input_b_slice = tl.load(input_b_ptr + offsets_in, mask=mask_in, other=0.0)
        
        # Load weight slice
        weights_slice = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE_N * BLOCK_SIZE_K), mask=tl.arange(0, BLOCK_SIZE_N * BLOCK_SIZE_K) < (out_features * in_features))
        
        # Matrix multiplication
        result = tl.dot(input_b_slice, weights_slice)
        tl.store(output_linear_ptr + offsets_out, result, mask=mask_out)

# Optimized kernel wrapper
@torch.fx.wrap
def optimized_linear_mul(in_0, in_1, in_2, in_3):
    # Get tensor shapes
    batch_size = in_2.size(0)
    seq_len = in_2.size(1) if in_2.dim() > 1 else 1
    in_features = in_2.size(-1)
    out_features = in_0.size(0)
    
    # Create output tensors
    output_mul = torch.empty_like(in_2)
    
    if in_3.dim() == 3:
        output_linear = torch.empty(size=(batch_size, seq_len, out_features), dtype=in_3.dtype, device=in_3.device)
    else:
        output_linear = torch.empty(size=(batch_size, out_features), dtype=in_3.dtype, device=in_3.device)
    
    # Tile size configuration
    BLOCK_SIZE_M = min(256, in_features)
    BLOCK_SIZE_N = min(64, out_features)
    BLOCK_SIZE_K = min(128, in_features)
    
    # Calculate grid size
    total_elements = batch_size * seq_len * max(in_features, out_features)
    grid_size = (total_elements + BLOCK_SIZE_M * BLOCK_SIZE_N * BLOCK_SIZE_K - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N * BLOCK_SIZE_K)
    
    # Launch kernel
    linear_mul_kernel[grid_size](
        in_0,
        in_1,
        in_2,
        in_3,
        output_mul,
        output_linear,
        batch_size,
        seq_len, 
        in_features,
        out_features,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return (output_mul, output_linear)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_linear_mul