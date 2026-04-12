import torch
import triton
import triton.language as tl

# Pattern: torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
def pattern(x, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr,           # Input tensor pointer
    weight_ptr,      # Weight tensor pointer
    bias_ptr,        # Bias tensor pointer
    out_ptr,         # Output tensor pointer
    n_features,      # Number of features (768)
    eps,             # Epsilon value
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= n_features:
        return
    
    # For layer normalization with one batch element [1, n_features],
    # compute mean and variance over all features
    if pid == 0:
        # Master thread computes global statistics
        sum_x = 0.0
        sum_x2 = 0.0
        
        # Compute sums over all features
        for i in range(n_features):
            x_i = tl.load(x_ptr + i)
            sum_x += x_i
            sum_x2 += x_i * x_i
        
        # Compute mean and variance
        mean = sum_x / n_features
        var = (sum_x2 / n_features) - (mean * mean)
        
        # Apply normalization and weight/bias to all elements
        for i in range(n_features):
            x_i = tl.load(x_ptr + i)
            w_i = tl.load(weight_ptr + i)
            b_i = tl.load(bias_ptr + i)
            
            normalized = (x_i - mean) / tl.sqrt(var + eps)
            out_val = normalized * w_i + b_i
            
            tl.store(out_ptr + i, out_val)

@torch.fx.wrap
def triton_layer_norm(x, normalized_shape, weight, bias, eps):
    n_features = normalized_shape[0]  # 768 in our case
    
    out = torch.empty_like(x)
    
    # Launch kernel with one program
    layer_norm_kernel[(1,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_features=n_features,
        eps=eps,
        BLOCK_SIZE=1,
    )
    
    return out

def replacement_func():
    return triton_layer_norm