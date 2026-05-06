import torch
import triton
import triton.language as tl


def pattern(input, weight, bias, layer_norm_bias, layer_norm_weight, eps=1e-05):
    conv_out = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    layer_norm_out = torch.nn.functional.layer_norm(
        conv_out, 
        (weight.shape[0], 1, 1), 
        layer_norm_bias, 
        layer_norm_weight, 
        eps
    )
    return torch.nn.functional.relu(layer_norm_out, inplace=True)

def replacement_args(input, weight, bias, layer_norm_bias, layer_norm_weight, eps=1e-05):
    return (input, weight, bias, layer_norm_bias, layer_norm_weight, eps)

def layer_norm_conv1x1_kernel(input_ptr, layer_norm_bias_ptr, layer_norm_weight_ptr, out_ptr, N, C, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, N)
    mask = tl.arange(0, BLOCK_SIZE) < (end - start)
    
    # Load input
    x = tl.load(input_ptr + start, mask=mask, other=0.0)
    
    # Calculate mean
    mean = tl.sum(x, axis=0) / tl.float32(BLOCK_SIZE)
    
    # Calculate variance
    var = tl.sum((x - mean) ** 2, axis=0)
    
    # Normalize
    normalized = (x - mean) / tl.sqrt(var + eps)
    
    # Apply layer norm scale and shift
    layer_norm_weight = tl.load(layer_norm_weight_ptr)
    layer_norm_bias = tl.load(layer_norm_bias_ptr)
    normalized = layer_norm_weight * normalized + layer_norm_bias
    
    # Store result
    tl.store(out_ptr + start, normalized, mask=mask)

torch.fx.wrap
@torch.fx.wrap
def layer_norm_conv1x1(input, weight, bias, layer_norm_bias, layer_norm_weight, eps=1e-05):
    N = input.numel()
    C = weight.shape[0]
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(input)
    
    layer_norm_conv1x1_kernel[  (num_programs,)  ](
        input_ptr=input,
        layer_norm_bias_ptr=layer_norm_bias,
        layer_norm_weight_ptr=layer_norm_weight,
        out_ptr=out,
        N=N,
        C=C,
        eps=eps,
        BLOCK_SIZE=128,
    )
    return out

def replacement_func():
    return layer_norm_conv1x1