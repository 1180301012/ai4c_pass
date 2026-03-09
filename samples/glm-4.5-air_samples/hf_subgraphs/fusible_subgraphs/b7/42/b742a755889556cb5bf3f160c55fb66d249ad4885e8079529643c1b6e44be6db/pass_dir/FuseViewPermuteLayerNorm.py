import torch
import triton
import triton.language as tl

def pattern(tmp_5, tmp_1, tmp_0):
    # Match the sequence exactly as in the model:
    # tmp_5.view(1, 384, 576) -> permute(0, 2, 1) -> layer_norm with tmp_1, tmp_0
    tmp_6 = tmp_5.view(1, 384, 576)  # [1, 384, 576]
    tmp_7 = tmp_6.permute(0, 2, 1)   # [1, 576, 384]
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (384,), tmp_1, tmp_0, 1e-05)  # [1, 576, 384]
    return tmp_8

def replacement_args(tmp_5, tmp_1, tmp_0):
    return (tmp_5, tmp_1, tmp_0)

@triton.jit
def layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, 
                      n_features, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute normalization (we'll handle the more complex case with mean/var in wrapper)
    mean = tl.sum(x) / n_features
    var = tl.sum((x - mean) * (x - mean)) / n_features
    x_normalized = (x - mean) * tl.sqrt(var + 1e-05)
    
    # Apply weight and bias
    weight = tl.load(weight_ptr + (offsets % n_features), mask=mask, other=0.0)
    bias = tl.load(bias_ptr + (offsets % n_features), mask=mask, other=0.0)
    out = x_normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_view_permute_layer_norm(x, weight, bias):
    # Original computation shapes
    x_reshaped = x.view(1, 384, 576)  # [1, 384, 576] -> [1, 576, 384]
    x_permuted = x_reshaped.permute(0, 2, 1)  # [1, 576, 384]
    
    N, H, C = x_permuted.shape  # N=1, H=576, C=384
    n_elements = N * H * C
    n_features = C
    
    out = torch.empty_like(x_permuted)
    
    blockSize = 1024
    grid = (triton.cdiv(n_elements, blockSize),)
    
    layer_norm_kernel[grid](
        x_ptr=x_permuted,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_features=n_features,
        n_elements=n_elements,
        BLOCK_SIZE=blockSize
    )
    
    return out

def replacement_func():
    return fused_view_permute_layer_norm