import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_seq,
    n_features,
    eps: tl.constexpr,
):
    # Each block handles one sequence
    seq_id = tl.program_id(0)
    start = seq_id * n_features
    # Load the entire sequence into registers (n_features=384 is small)
    x = tl.load(x_ptr + start, mask=tl.arange(0, n_features) < n_features)
    
    # Compute mean across features (dim=2)
    mean = tl.sum(x, axis=0) / n_features
    
    # Compute variance across features
    var = tl.sum((x - mean) ** 2, axis=0) / n_features + eps
    
    # Apply LayerNorm: (x - mean) / sqrt(var) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var)
    x_norm = x_norm * tl.load(weight_ptr + tl.arange(0, n_features)) + tl.load(bias_ptr + tl.arange(0, n_features))
    
    # Store normalized sequence
    tl.store(out_ptr + start, x_norm)

@torch.fx.wrap
def layer_norm_fused(x, weight, bias, eps=1e-12):
    batch, seq, feature = x.shape
    n_seq = seq
    n_features = feature
    out = torch.empty_like(x)
    grid = (n_seq,)
    layer_norm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_seq=n_seq,
        n_features=n_features,
        eps=eps
    )
    return out

def pattern(in_5, in_6, in_1, in_2):
    # Match the exact computation: add followed by LayerNorm
    tmp_5 = in_6 + in_5
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    return tmp_6

def replacement_args(in_5, in_6, in_1, in_2):
    # Return inputs for the fused kernel: sum + LayerNorm params
    return (in_6 + in_5, in_1, in_2)

def replacement_func():
    return layer_norm_fused