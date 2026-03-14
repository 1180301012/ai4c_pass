import torch
import triton
import triton.language as tl

def pattern(input_tensor, normalized_shape, weight, bias, eps):
    """
    Pattern for layer normalization
    """
    output = torch.nn.functional.layer_norm(input_tensor, normalized_shape, weight, bias, eps)
    return output

def replacement_args(input_tensor, normalized_shape, weight, bias, eps):
    return (input_tensor, weight, bias, eps)

@triton.jit
def layer_norm_forward_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    N, D,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row = tl.program_id(0)
    
    if row >= N:
        return
    
    # Compute mean
    mean = 0.0
    for col in range(0, D, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
        mean += tl.sum(x)
    mean = mean / D
    
    # Compute variance
    var = 0.0
    for col in range(0, D, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
        diff = x - mean
        var += tl.sum(diff * diff)
    var = var / D
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply affine transform
    for col in range(0, D, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        
        x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
        bias_val = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        
        x_hat = (x - mean) * rstd
        out = x_hat * weight + bias_val
        
        tl.store(output_ptr + row * D + cols, out, mask=mask)

@torch.fx.wrap
def triton_layer_norm(input_tensor, weight, bias, eps):
    # Flatten to 2D
    original_shape = input_tensor.shape
    x_2d = input_tensor.reshape(-1, original_shape[-1])
    
    N, D = x_2d.shape
    output = torch.empty_like(x_2d)
    
    BLOCK_SIZE = triton.next_power_of_2(min(D, 1024))
    grid = (N,)
    
    layer_norm_forward_kernel[grid](
        x_2d, weight, bias, output,
        N, D,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.reshape(original_shape)

def replacement_func():
    return triton_layer_norm