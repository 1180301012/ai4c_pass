import torch
import triton
import triton.language as tl

# Pattern matching for the linear operation
# Matches torch.nn.functional.linear(input, weight, bias)
def pattern(input_tensor, weight_tensor, bias_tensor):
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)

# Extract arguments for replacement
# Returns (input, weight, bias) as they are consumed in the operation
def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

# Triton kernel for optimized linear operation
@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_features, out_features,
    BLOCK_SIZE: tl.constexpr = 128
):
    # Each block handles one output feature and one batch element
    out_feat = tl.program_id(0)
    batch_idx = tl.program_id(1)

    # Load bias for this output feature
    bias_val = tl.load(bias_ptr + out_feat)

    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)

    # Process input features in blocks for vectorized computation
    for i in range(0, in_features, BLOCK_SIZE):
        # Load input segment (for current batch) and weight segment (for current out_feat)
        input_segment = tl.load(
            input_ptr + batch_idx * in_features + i,
            mask=i + tl.arange(0, BLOCK_SIZE) < in_features,
            other=0.0
        )
        weight_segment = tl.load(
            weight_ptr + out_feat * in_features + i,
            mask=i + tl.arange(0, BLOCK_SIZE) < in_features,
            other=0.0
        )

        # Dot product accumulation
        acc += tl.sum(input_segment * weight_segment)

    # Add bias and store result
    acc = acc + bias_val
    tl.store(output_ptr + batch_idx * out_features + out_feat, acc)

# Triton kernel wrapper
@torch.fx.wrap
def linear_wrapper(input_tensor, weight_tensor, bias_tensor):
    # Determine tensor shapes
    batch_size = input_tensor.shape[0]
    in_features = input_tensor.shape[1]
    out_features = weight_tensor.shape[0]

    # Create output tensor
    output = torch.empty((batch_size, out_features), dtype=input_tensor.dtype, device=input_tensor.device)

    # Set kernel grid dimensions
    grid = (out_features, batch_size)

    # Launch kernel
    linear_kernel[grid](
        input_tensor, weight_tensor, bias_tensor, output,
        batch_size, in_features, out_features,
        BLOCK_SIZE=128
    )

    return output

# Replacement function (returns the optimized wrapper)
def replacement_func():
    return linear_wrapper