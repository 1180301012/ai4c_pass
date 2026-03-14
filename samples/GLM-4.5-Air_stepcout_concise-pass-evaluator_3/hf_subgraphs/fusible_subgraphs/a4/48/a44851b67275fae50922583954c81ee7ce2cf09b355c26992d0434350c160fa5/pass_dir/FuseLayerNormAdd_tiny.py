import torch
import triton
import triton.language as tl

# Pattern matching function for layer norm + addition (tiny model)
def pattern(in_2, in_3, tmp_1, tmp_0):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (192,), tmp_1, tmp_0, 1e-06)
    return tmp_2, tmp_3

# Argument extraction function
def replacement_args(in_2, in_3, tmp_1, tmp_0):
    return (in_2, in_3, tmp_1, tmp_0)

# Triton kernel for fused add + layer norm (tiny model)
@triton.jit
def fused_norm_kernel(
    x1_ptr, x2_ptr, 
    weight_ptr, bias_ptr, out_ptr,
    n_tokens, n_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one token
    token_idx = tl.program_id(0)
    
    if token_idx >= n_tokens:
        return
    
    # Calculate pointers for this token
    feature_offset = token_idx * n_features
    
    # Load input features for this token
    x1 = tl.load(x1_ptr + feature_offset, mask=feature_offset + tl.arange(0, BLOCK_SIZE) < n_tokens * n_features, other=0.0)
    x2 = tl.load(x2_ptr + feature_offset, mask=feature_offset + tl.arange(0, BLOCK_SIZE) < n_tokens * n_features, other=0.0)
    
    # Addition
    x = x1 + x2
    
    # Load weight and bias (broadcasted across tokens)
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr)
    
    # Layer normalization computation
    mean = tl.sum(x) / n_features
    var = tl.sum((x - mean) * (x - mean)) / n_features
    std = tl.sqrt(var + 1e-06)
    
    # Normalize and apply scale/shift
    out = (x - mean) / std * weight + bias
    
    # Store result
    tl.store(out_ptr + feature_offset, out, mask=feature_offset + tl.arange(0, BLOCK_SIZE) < n_tokens * n_features)

@torch.fx.wrap
def fused_layer_norm_add(x1, x2, weight, bias):
    # x1, x2 shape: [1, 196, 192]
    # weight shape: [192]
    # bias shape: [192]
    
    n_tokens = x1.shape[1]  # 196
    n_features = x1.shape[2]  # 192
    
    out = torch.empty_like(x1)
    
    BLOCK_SIZE = 1024
    num_programs = (n_tokens * n_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_norm_kernel[(num_programs,)](
        x1=x1,
        x2=x2, 
        weight=weight,
        bias=bias,
        out=out,
        n_tokens=n_tokens,
        n_features=n_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out, x1 + x2  # Return both intermediate and final result

# Replacement function (returns function reference)
def replacement_func():
    return fused_layer_norm_add