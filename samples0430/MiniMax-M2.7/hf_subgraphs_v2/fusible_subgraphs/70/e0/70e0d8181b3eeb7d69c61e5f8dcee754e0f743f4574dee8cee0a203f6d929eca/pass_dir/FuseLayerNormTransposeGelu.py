import torch
import triton
import triton.language as tl

# Pattern matching function - matches the LayerNorm + Transpose + GELU pattern
def pattern(bias, weight, x):
    """
    Match the pattern: layer_norm + transpose(-2, -1) + gelu
    Args:
        bias: LayerNorm bias tensor [512]
        weight: LayerNorm weight tensor [512]
        x: Input tensor [1, 3999, 512]
    Returns:
        tmp_4: Result of layer_norm → transpose(-2, -1) → gelu
    """
    # LayerNorm with normalized_shape=(512,), weight, bias, eps=1e-05
    tmp_2 = torch.nn.functional.layer_norm(x, (512,), weight, bias, 1e-05)
    # Transpose last two dimensions
    tmp_3 = tmp_2.transpose(-2, -1)
    # GELU activation
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(bias, weight, x):
    """Extract arguments needed for the replacement function"""
    return (bias, weight, x)


@triton.jit
def layer_norm_gelu_transpose_kernel(
    # Input pointers
    x_ptr, weight_ptr, bias_ptr,
    # Output pointer
    out_ptr,
    # Strides
    x_stride_0, x_stride_1, x_stride_2,
    out_stride_0, out_stride_1, out_stride_2,
    # Shape info
    batch_size, seq_len, hidden_size,
    # Normalization parameters
    eps: tl.constexpr,
):
    """
    Fused kernel for LayerNorm + GELU + Transpose
    
    Input x: [batch_size, seq_len, hidden_size] = [1, 3999, 512]
    Output: [batch_size, hidden_size, seq_len] = [1, 512, 3999]
    
    Grid: (batch_size * seq_len) = 3999 programs
    Each program processes one (batch, seq_idx) position
    """
    # Get program IDs
    batch_pid = tl.program_id(0) // seq_len
    seq_idx = tl.program_id(0) % seq_len
    
    # Compute offsets
    x_base = batch_pid * x_stride_0 + seq_idx * x_stride_1
    out_base = batch_pid * out_stride_0 + seq_idx * out_stride_2
    
    # Load all hidden values for this position
    x_vals = tl.load(x_ptr + x_base + tl.arange(0, 512) * x_stride_2).to(tl.float32)
    
    # Load weight and bias vectors
    w = tl.load(weight_ptr + tl.arange(0, 512)).to(tl.float32)
    b = tl.load(bias_ptr + tl.arange(0, 512)).to(tl.float32)
    
    # LayerNorm: compute mean and variance
    mean = tl.sum(x_vals) / 512.0
    var = tl.sum(x_vals * x_vals) / 512.0 - mean * mean
    inv_std = tl.rsqrt(var + eps)
    
    # Apply normalization with affine parameters
    norm = (x_vals - mean) * inv_std * w + b
    
    # GELU: use sigmoid-based tanh
    tanh_arg = 0.7978845608028654 * (norm + 0.04471497864598608 * norm * norm * norm)
    tanh_arg = tl.where(tanh_arg > 10.0, 10.0, tl.where(tanh_arg < -10.0, -10.0, tanh_arg))
    tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * tanh_arg)) - 1.0
    cdf = 0.5 * (1.0 + tanh_val)
    out_vals = norm * cdf
    
    # Store at transposed position
    tl.store(out_ptr + out_base + tl.arange(0, 512) * out_stride_1, out_vals)


@torch.fx.wrap
def triton_layer_norm_transpose_gelu(bias, weight, x):
    """
    Fully fused LayerNorm + Transpose + GELU using Triton kernel
    """
    batch_size, seq_len, hidden_size = x.shape
    
    # Allocate output tensor
    out = torch.empty((batch_size, hidden_size, seq_len), 
                      dtype=x.dtype, device=x.device)
    
    # Use simpler grid: one program per output element
    grid = (batch_size * seq_len,)
    
    layer_norm_gelu_transpose_kernel[grid](
        x, weight, bias, out,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        batch_size, seq_len, hidden_size,
        1e-05
    )
    
    return out


def replacement_func():
    """Returns the replacement function"""
    return triton_layer_norm_transpose_gelu