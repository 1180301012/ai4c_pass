import torch
import triton
import triton.language as tl


@triton.jit
def triton_batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr, output_ptr,
    batch_size, num_features, eps,
    stride_in,
):
    """
    Triton kernel for fused batch normalization (inference mode).
    Formula: output = (input - running_mean) / sqrt(running_var + eps) * weight + bias
    """
    # Each program handles a feature
    feat_idx = tl.program_id(0)
    
    # Load stats for this feature
    running_mean = tl.load(running_mean_ptr + feat_idx)
    running_var = tl.load(running_var_ptr + feat_idx)
    weight = tl.load(weight_ptr + feat_idx)
    bias = tl.load(bias_ptr + feat_idx)
    
    # Compute normalization factor: 1 / sqrt(var + eps)
    inv_std = tl.rsqrt(running_var + eps)
    
    # Process all batch elements for this feature
    for b in range(batch_size):
        input_offset = b * stride_in + feat_idx
        output_offset = b * num_features + feat_idx
        
        # Load, normalize, and store
        input_val = tl.load(input_ptr + input_offset)
        normalized = (input_val - running_mean) * inv_std * weight + bias
        tl.store(output_ptr + output_offset, normalized)


@torch.fx.wrap
def triton_batch_norm(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05):
    """
    Wrapper for the Triton batch normalization kernel.
    """
    batch_size, num_features = input.shape
    
    # Allocate output
    output = torch.empty_like(input)
    
    # Grid: one program per feature
    grid = (num_features,)
    
    # Launch kernel
    triton_batch_norm_kernel[grid](
        input, running_mean, running_var, weight, bias, output,
        batch_size, num_features, eps,
        input.stride(0)
    )
    
    return output


def pattern(in_7, in_0, in_1, in_3, in_2):
    """
    Pattern for torch.nn.functional.batch_norm (simplified - constant args)
    
    Args:
    - in_7: input tensor [batch, num_features]
    - in_0: running_mean [num_features]
    - in_1: running_var [num_features]
    - in_3: weight (gamma) [num_features]
    - in_2: bias (beta) [num_features]
    """
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_7


def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)


def replacement_func():
    return triton_batch_norm