import torch
import triton
import triton.language as tl

def pattern(tensor):
    """
    Match the pattern: compute l2 norm along last dimension and divide
    SPECIAL NOTE: The operations here exactly mirror the original PyTorch code for the normalization step.
    """
    norm = tensor.norm(p=2, dim=-1, keepdim=True)
    normalized = tensor / norm
    return normalized

def replacement_args(tensor):
    return (tensor,)

@triton.jit
def normalize_kernel(
    tensor_ptr,
    output_ptr,
    batch_size,
    feature_size,
    BLOCK_SIZE: tl.constexpr = 128,
):
    batch_id = tl.program_id(0)
    if batch_id >= batch_size:
        return

    # Initialize sum of squares
    sum_sq = tl.zeros(1, dtype=tl.float32)
    
    # Process each feature in batch
    for i in range(feature_size):
        idx = batch_id * feature_size + i
        val = tl.load(tensor_ptr + idx, mask=tl.ones_like(i), other=0.0)
        sum_sq += val * val

    # Compute L2 norm
    norm = tl.sqrt(sum_sq)

    # Normalize the features
    for i in range(feature_size):
        idx = batch_id * feature_size + i
        val = tl.load(tensor_ptr + idx, mask=tl.ones_like(i), other=0.0)
        tl.store(output_ptr + idx, val / norm, mask=tl.ones_like(i))

@torch.fx.wrap
def normalize_kernel_wrapper(tensor):
    batch_size = tensor.shape[0]
    feature_size = tensor.shape[1]
    output = torch.empty_like(tensor)
    num_programs = batch_size
    
    normalize_kernel[num_programs, 1](
        tensor_ptr=tensor,
        output_ptr=output,
        batch_size=batch_size,
        feature_size=feature_size,
        BLOCK_SIZE=128,
    )
    return output

def replacement_func():
    return normalize_kernel_wrapper