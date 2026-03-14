import torch
import triton
import triton.language as tl


def pattern(in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    """
    Pattern to match: torch.nn.functional.batch_norm(in_7, running_mean, running_var, weight, bias, training, momentum, eps)
    
    When training=False:
    - in_7: input tensor [batch, features]
    - tmp_0: running_mean [features]
    - tmp_1: running_var [features] 
    - tmp_3: weight [features]
    - tmp_2: bias [features]
    - False: training mode (False)
    - 0.1: momentum
    - 1e-05: eps
    """
    tmp_7 = torch.nn.functional.batch_norm(in_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    return tmp_7


def replacement_args(in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    return (in_7, tmp_0, tmp_1, tmp_3, tmp_2)


# Simple batch norm kernel - one program per feature, process all batch elements
@triton.jit
def triton_batch_norm_kernel(
    input_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    num_elements, num_features,
    eps: tl.constexpr,
):
    """
    Triton kernel for batch normalization inference.
    y = (x - mean) / sqrt(var + eps) * weight + bias
    
    Each program processes one feature across all batch elements.
    """
    # Block size
    BLOCK_SIZE: tl.constexpr = 64
    
    # Get feature index
    fid = tl.program_id(0)
    
    # Compute the offset for this feature
    # Each program handles one feature (fid), but we use BLOCK_SIZE for the batch loop
    feature_offsets = tl.arange(0, BLOCK_SIZE)
    feature_mask = feature_offsets < num_features
    
    # Load parameters for this feature (just the first element since all are the same for a feature)
    mean = tl.load(mean_ptr + fid)
    var = tl.load(var_ptr + fid)
    weight = tl.load(weight_ptr + fid)
    bias = tl.load(bias_ptr + fid)
    
    # Compute normalization parameters
    std = tl.sqrt(var + eps)
    norm_factor = weight / std
    mean_term = -mean * norm_factor + bias
    
    # Process all batch elements
    batch_offsets = tl.arange(0, BLOCK_SIZE)
    
    for batch_start in range(0, num_elements, BLOCK_SIZE):
        batch_idx = batch_start + batch_offsets
        batch_mask = batch_idx < num_elements
        
        # Create combined mask for this iteration
        # We need to load input[batch_idx, feature=fid]
        # The input is stored as [batch, features], so we index as: batch_idx * num_features + fid
        load_ptrs = input_ptr + batch_idx * num_features + fid
        x = tl.load(load_ptrs, mask=batch_mask, other=0.0)
        
        # Apply normalization: x * norm_factor + mean_term
        output = x * norm_factor + mean_term
        
        # Store output
        store_ptrs = output_ptr + batch_idx * num_features + fid
        tl.store(store_ptrs, output, mask=batch_mask)


@torch.fx.wrap
def triton_batch_norm_inference(input, running_mean, running_var, weight, bias, eps=1e-05):
    """
    Optimized batch normalization for inference using Triton.
    """
    num_elements = input.shape[0]  # batch size
    num_features = input.shape[1]  # number of features
    
    # Allocate output
    output = torch.empty_like(input)
    
    # Grid: one program per feature channel
    grid = (num_features,)
    
    # Launch kernel
    triton_batch_norm_kernel[grid](
        input, output,
        running_mean, running_var, weight, bias,
        num_elements, num_features,
        eps,
    )
    
    return output


def replacement_func():
    return triton_batch_norm_inference