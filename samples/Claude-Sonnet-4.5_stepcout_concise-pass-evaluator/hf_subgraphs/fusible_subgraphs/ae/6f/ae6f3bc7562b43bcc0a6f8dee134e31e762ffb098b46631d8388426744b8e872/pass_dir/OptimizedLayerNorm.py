import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    hidden_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one token in batch x seq_len)
    row_idx = tl.program_id(0)
    
    if row_idx >= N:
        return
    
    # Calculate mean
    mean = 0.0
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        row_offset = row_idx * hidden_dim + offsets
        vals = tl.load(input_ptr + row_offset, mask=mask, other=0.0)
        mean += tl.sum(vals, axis=0)
    mean = mean / hidden_dim
    
    # Calculate variance
    var = 0.0
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        row_offset = row_idx * hidden_dim + offsets
        vals = tl.load(input_ptr + row_offset, mask=mask, other=0.0)
        diff = vals - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / hidden_dim
    
    # Normalize and apply affine transformation
    rstd = 1.0 / tl.sqrt(var + eps)
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        row_offset = row_idx * hidden_dim + offsets
        
        vals = tl.load(input_ptr + row_offset, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        normalized = (vals - mean) * rstd
        output = normalized * weight + bias
        
        tl.store(output_ptr + row_offset, output, mask=mask)


@torch.fx.wrap
def triton_layernorm(input_tensor, normalized_shape, weight, bias, eps):
    # Get dimensions
    batch_size, seq_len, hidden_dim = input_tensor.shape
    N = batch_size * seq_len
    
    # Flatten batch and seq dimensions
    input_flat = input_tensor.view(N, hidden_dim)
    output = torch.empty_like(input_flat)
    
    BLOCK_SIZE = 256
    grid = (N,)
    
    layernorm_kernel[grid](
        input_flat,
        weight,
        bias,
        output,
        N,
        hidden_dim,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.view(batch_size, seq_len, hidden_dim)


def pattern(input_tensor, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(input_tensor, normalized_shape, weight, bias, eps)


def replacement_args(input_tensor, normalized_shape, weight, bias, eps):
    return (input_tensor, normalized_shape, weight, bias, eps)


def replacement_func():
    return triton_layernorm