import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(tmp_2, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(tmp_2, normalized_shape, weight, bias, eps)

# Argument extraction function
def replacement_args(tmp_2, normalized_shape, weight, bias, eps):
    return (tmp_2, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr, 
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    hidden_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sequence element
    seq_idx = tl.program_id(0)
    seq_start = seq_idx * hidden_dim

    # Shared memory for reduction
    shmem_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    shmem_x2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Threads load elements of the hidden dimension
    # Replace with new kernel logic that doesn't require thread_id
    seq_idx = tl.program_id(0)
    seq_start = seq_idx * hidden_dim

    # Each thread in the block handles a portion of the hidden dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # Load elements for this sequence element
    x_vals = tl.load(x_ptr + seq_start + offsets, mask=mask)

    # Calculate sum and sum of squares
    sum_x = tl.sum(x_vals, axis=0)
    sum_x2 = tl.sum(x_vals * x_vals, axis=0)

    # Calculate mean and variance
    mean = sum_x / hidden_dim
    var = sum_x2 / hidden_dim - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply weight/bias
    x_norm = (x_vals - mean) * inv_std
    weights = tl.load(weight_ptr + offsets, mask=mask)
    biases = tl.load(bias_ptr + offsets, mask=mask)
    out = x_norm * weights + biases

    # Store result
    tl.store(out_ptr + seq_start + offsets, out, mask=mask)



    # Calculate mean and variance
    mean = sum_x / hidden_dim
    var = sum_x2 / hidden_dim - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply weight/bias

@torch.fx.wrap
def optimized_layer_norm(x, normalized_shape, weight, bias, eps):
    # Extract hidden dimension from normalized_shape
    hidden_dim = normalized_shape[0]
    n_elements = x.numel()
    num_seq = n_elements // hidden_dim

    # Create output tensor
    out = torch.empty_like(x)

    # Configure kernel
    BLOCK_SIZE = 128
    grid = (num_seq,)

    # Launch kernel
    layer_norm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        hidden_dim=hidden_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return optimized_layer_norm