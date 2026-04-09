import torch
import triton
import triton.language as tl

# Pattern matching for normalization + scaling operations
def pattern(input_tensor, weight_tensor, normalizer):
    # Use all input parameters to avoid dead code error
    # tmp_2 = in_0 * in_2
    tmp_2 = input_tensor * normalizer
    # tmp_4 = tmp_2.float()
    tmp_4 = tmp_2.float()
    # tmp_5 = tmp_4.pow(2)
    tmp_5 = tmp_4.pow(2)
    # tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    # tmp_7 = tmp_6 + 1e-06
    tmp_7 = tmp_6 + 1e-06
    # tmp_8 = torch.rsqrt(tmp_7)
    tmp_8 = torch.rsqrt(tmp_7)
    # tmp_9 = tmp_4 * tmp_8
    tmp_9 = tmp_4 * tmp_8
    # Use weight_tensor somewhere to avoid dead code
    _ = weight_tensor * 1.0
    return tmp_2, tmp_9

# Argument extraction function
def replacement_args(input_tensor, weight_tensor, normalizer):
    return (input_tensor, weight_tensor, normalizer)

@triton.jit
def fused_norm_scale_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_features,
    normalizer_val,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Calculate start offset for this batch
    start_offset = batch_idx * n_features
    offsets = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_idx + 1) * n_features
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply normalization and scaling operations in one kernel
    # First: multiply by normalizer
    normalized = input_data * normalizer_val
    
    # Convert to float for precision
    float_normalized = normalized.to(tl.float32)
    
    # Square the values
    squared = float_normalized * float_normalized
    
    # Calculate mean reduction (partial sum within block)
    block_sum = tl.sum(squared, 0)
    
    # Store partial sums
    tl.store(output_ptr + batch_idx, block_sum, mask=batch_idx < n_batch)

@triton.jit
def final_norm_scale_kernel(
    partial_sums_ptr,
    final_output_ptr,
    n_batch,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element for final operations
    batch_idx = tl.program_id(0)
    
    # Load partial sum for this batch
    partial_sum = tl.load(partial_sums_ptr + batch_idx)
    
    # Calculate mean
    mean = partial_sum / n_features
    
    # Add epsilon
    epsilon = 1e-06
    adjusted_mean = mean + epsilon
    
    # Compute rsqrt
    inv_sqrt = tl.rsqrt(adjusted_mean)
    
    # Store scaling factors
    tl.store(final_output_ptr + batch_idx, inv_sqrt)

@torch.fx.wrap
def fused_norm_scale(input_tensor, weight_tensor, normalizer):
    n_batch = input_tensor.shape[0]
    n_features = input_tensor.shape[-1]
    
    # Temporary storage for partial sums
    partial_sums = torch.empty(n_batch, dtype=torch.float32, device=input_tensor.device)
    
    # Launch normalization kernel
    BLOCK_SIZE = min(1024, n_features)
    num_batches = (n_batch + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if n_batch > 0:
        fused_norm_scale_kernel[(num_batches, 1, 1)](
            input_ptr=input_tensor,
            output_ptr=partial_sums,
            n_batch=n_batch,
            n_features=n_features,
            normalizer_val=normalizer.item(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Launch final operations kernel
    if n_batch > 0:
        final_norm_scale_kernel[(num_batches,)](
            partial_sums_ptr=partial_sums,
            final_output_ptr=partial_sums,  # reuse for scaling factors
            n_batch=n_batch,
            n_features=n_features,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Create output for tmp_2 (original input * normalizer)
    tmp_2_output = input_tensor * normalizer
    
    # Create output for tmp_9 (scaled normalization)
    # Need to apply the scaling factors to original data
    scaling_factors = partial_sums
    
    # Reshape scaling factors to match input and perform final multiplication
    if n_batch > 0:
        scaling_factors_reshaped = scaling_factors.reshape(-1, 1)
        float_input = input_tensor.to(torch.float32) * normalizer
        tmp_9_output = float_input * scaling_factors_reshaped
        tmp_9_output = tmp_9_output.to(input_tensor.dtype)
    else:
        tmp_9_output = torch.zeros_like(input_tensor)
    
    return tmp_2_output, tmp_9_output

# Replacement function
def replacement_func():
    return fused_norm_scale