import torch
import triton
import triton.language as tl

# Pattern for Tree-ConditionHK: embed_dim=128, H=W=56
def pattern(input_tensor, weight, bias):
    tmp_12 = torch.nn.functional.layer_norm(input_tensor, (128,), weight, bias, 1e-05)
    tmp_13 = tmp_12.view(1, 56, 56, 128)
    return tmp_13

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def layernorm_kernel_128_56(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Load input row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < C
    
    row_ptr = input_ptr + row_idx * C
    x = tl.load(row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Layer norm computation
    mean = tl.sum(x, axis=0) / C
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd
    
    # Load weight and bias
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Apply affine transformation
    y = x_norm * w + b
    
    # Store to output
    out_ptr = output_ptr + row_idx * C
    tl.store(out_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def layernorm_view_128_56(input_tensor, weight, bias):
    H = 56
    W = 56
    C = 128
    
    N = H * W
    
    # Squeeze batch dimension if needed (input is [1, seq_len, C])
    input_2d = input_tensor.view(-1, C)
    
    # Output shape: [1, H, W, C]
    output = torch.empty(N, C, device=input_tensor.device, dtype=input_tensor.dtype)
    
    layernorm_kernel_128_56[(N,)](
        input_2d, weight, bias, output,
        N=N, C=C,
        eps=1e-05,
    )
    
    # Reshape to [1, H, W, C]
    return output.view(1, H, W, C)

def replacement_func():
    return layernorm_view_128_56