import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire LayerNorm computation chain
def pattern(input_3, input_2, in_1, in_0):
    # Input addition
    tmp_3 = input_3 + input_2
    # Type conversion to float32 for numerical precision
    tmp_4 = tmp_3.float()
    # Mean calculation over last dimension
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    # Compute variance: E[X^2] - (E[X])^2
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    # Standard deviation with epsilon for numerical stability
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    # Standardization: (X - mean) / std
    tmp_9 = tmp_4 - tmp_5  # Reuse tmp_4 - tmp_5 computation
    tmp_12 = tmp_9 / tmp_11
    # Convert explicitly to float32 for consistent multiplication
    tmp_13 = tmp_12.to(torch.float32)
    # Apply scaling and shifting parameters
    tmp_14 = in_1 * tmp_13
    # Residual connection
    tmp_15 = tmp_14 + in_0
    return tmp_15

# Argument extraction function
def replacement_args(input_3, input_2, in_1, in_0):
    return (input_3, input_2, in_1, in_0)

# Triton kernel for fused LayerNorm with scaling/shifting
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_HIDDEN": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_HIDDEN": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE_HIDDEN": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE_HIDDEN": 256}, num_warps=8),
    ],
    key=['hidden_size'],
)
@triton.jit
def fused_layernorm_kernel(
    # Input tensors (broadcastable shapes)
    input_ptr,
    residual_ptr,
    scale_ptr,
    bias_ptr,
    output_ptr,
    # Original data type information
    orig_dtype: tl.constexpr,
    # Tensor shapes and strides
    batch_size,
    seq_len,
    hidden_size,
    # Strides for each dimension
    input_batch_stride,
    input_seq_stride,
    input_hidden_stride,
    residual_batch_stride,
    residual_seq_stride,
    residual_hidden_stride,
    scale_hidden_stride,
    bias_hidden_stride,
    output_batch_stride,
    output_seq_stride,
    output_hidden_stride,
    # Kernel configuration
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    # Calculate program IDs for batch and sequence dimensions
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Output pointer for current batch/sequence position
    output_base = output_ptr + batch_idx * output_batch_stride + seq_idx * output_seq_stride
    input_base = input_ptr + batch_idx * input_batch_stride + seq_idx * input_seq_stride
    residual_base = residual_ptr + batch_idx * residual_batch_stride + seq_idx * residual_seq_stride
    
    # Initialize accumulators for mean and variance
    mean_acc = 0.0
    var_acc = 0.0
    
    # First pass: compute mean
    for hidden_offset in range(0, hidden_size, BLOCK_SIZE_HIDDEN):
        # Compute bounds for current block
        end_offset = min(hidden_offset + BLOCK_SIZE_HIDDEN, hidden_size)
        current_block_size = end_offset - hidden_offset
        
        # Load input block
        input_offsets = hidden_offset + tl.arange(0, BLOCK_SIZE_HIDDEN)
        input_mask = input_offsets < hidden_size
        x = tl.load(input_base + input_offsets, mask=input_mask, other=0.0).to(tl.float32)
        
        # Accumulate sum for mean
        block_sum = tl.sum(x)
        mean_acc += block_sum
    
    # Compute mean across all blocks
    mean_acc = mean_acc / hidden_size
    
    # Second pass: compute variance
    for hidden_offset in range(0, hidden_size, BLOCK_SIZE_HIDDEN):
        end_offset = min(hidden_offset + BLOCK_SIZE_HIDDEN, hidden_size)
        current_block_size = end_offset - hidden_offset
        
        input_offsets = hidden_offset + tl.arange(0, BLOCK_SIZE_HIDDEN)
        input_mask = input_offsets < hidden_size
        x = tl.load(input_base + input_offsets, mask=input_mask, other=0.0).to(tl.float32)
        
        # Compute squared deviation from mean
        x_centered = x - mean_acc
        x_squared = x_centered * x_centered
        block_var = tl.sum(x_squared)
        var_acc += block_var
    
    # Compute final variance and standard deviation
    variance = var_acc / hidden_size
    std = tl.sqrt(variance + 1e-07)
    
    # Final pass: normalize, scale, shift, and add residual
    for hidden_offset in range(0, hidden_size, BLOCK_SIZE_HIDDEN):
        end_offset = min(hidden_offset + BLOCK_SIZE_HIDDEN, hidden_size)
        current_block_size = end_offset - hidden_offset
        
        input_offsets = hidden_offset + tl.arange(0, BLOCK_SIZE_HIDDEN)
        input_mask = input_offsets < hidden_size
        
        # Load data
        x = tl.load(input_base + input_offsets, mask=input_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_base + input_offsets, mask=input_mask, other=0.0)
        scale = tl.load(scale_ptr + input_offsets, mask=input_mask, other=1.0)
        bias = tl.load(bias_ptr + input_offsets, mask=input_mask, other=0.0)
        
        # Compute normalized value: (x - mean) / std * scale + bias + residual
        x_normalized = (x - mean_acc) / std
        x_scaled = x_normalized * scale + bias
        output_val = x_scaled + residual
        
        # Store result in original dtype based on tl.constexpr orig_dtype
        if orig_dtype == tl.bfloat16:
            output_val_orig = output_val.to(tl.bfloat16)
        else:
            output_val_orig = output_val.to(tl.float16)
        
        tl.store(output_base + input_offsets, output_val_orig, mask=input_mask)
        
        

# Kernel wrapper that handles launch configuration
@torch.fx.wrap
def fused_layernorm_forward(input_tensor, residual_tensor, scale_tensor, bias_tensor):
    # Get tensor shapes
    batch_size, seq_len, hidden_size = input_tensor.shape
    
    # Fixed block size for sequence dimension, hidden size will be autotuned
    BLOCK_SIZE_SEQ = 1
    
    # Calculate grid dimensions - BLOCK_SIZE_HIDDEN will be determined by autotuning
    grid = (
        batch_size,
        seq_len,
        (hidden_size + 64 - 1) // 64,  # Default 64 blocks, will be adjusted by autotuner
    )
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Map torch dtype to Triton dtype
    dtype_to_tl = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
        torch.float64: tl.float64
    }
    orig_dtype = dtype_to_tl.get(input_tensor.dtype, tl.float32)
    
    # Launch kernel
    fused_layernorm_kernel[grid](
        input_ptr=input_tensor,
        residual_ptr=residual_tensor,
        scale_ptr=scale_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        orig_dtype=orig_dtype,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        input_batch_stride=input_tensor.stride(0),
        input_seq_stride=input_tensor.stride(1),
        input_hidden_stride=input_tensor.stride(2),
        residual_batch_stride=residual_tensor.stride(0),
        residual_seq_stride=residual_tensor.stride(1),
        residual_hidden_stride=residual_tensor.stride(2),
        scale_hidden_stride=scale_tensor.stride(0) if len(scale_tensor.shape) > 0 else 1,
        bias_hidden_stride=bias_tensor.stride(0) if len(bias_tensor.shape) > 0 else 1,
        output_batch_stride=output.stride(0),
        output_seq_stride=output.stride(1),
        output_hidden_stride=output.stride(2),
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
    )
    
    return output

# Replacement function that returns the kernel wrapper
def replacement_func():
    return fused_layernorm_forward