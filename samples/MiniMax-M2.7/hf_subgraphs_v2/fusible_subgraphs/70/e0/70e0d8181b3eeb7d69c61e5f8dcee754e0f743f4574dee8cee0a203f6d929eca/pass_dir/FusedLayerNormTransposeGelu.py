import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_fused_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    n_elements,
    n_features: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm + Transpose + GELU kernel.
    Input shape: [batch, seq_len, n_features]
    Output shape: [batch, n_features, seq_len]
    
    The fusion applies:
    1. LayerNorm along the last dimension (n_features=512)
    2. Transpose (-2, -1): [batch, seq_len, n_features] -> [batch, n_features, seq_len]
    3. GELU activation
    """
    # Get position identifiers
    batch_seq_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Block size for computing reduction
    block_start = batch_seq_idx * n_features + feature_idx
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * n_features
    mask = block_start + tl.arange(0, BLOCK_SIZE) < batch_seq_idx * n_features + n_elements
    mask = mask & (block_start < (batch_seq_idx + 1) * n_features)
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + feature_idx).to(tl.float32)
    b = tl.load(bias_ptr + feature_idx).to(tl.float32)
    
    # Step 1: Compute mean for layer norm
    mean = 0.0
    for i in range(BLOCK_SIZE):
        offs = block_start + i * n_features
        m = offs < batch_seq_idx * n_features + n_elements
        if m:
            mean += tl.load(x_ptr + offs, mask=m, other=0.0)
    mean = mean / n_features
    
    # Step 2: Compute variance
    var = 0.0
    for i in range(BLOCK_SIZE):
        offs = block_start + i * n_features
        m = offs < batch_seq_idx * n_features + n_elements
        if m:
            diff = tl.load(x_ptr + offs, mask=m, other=0.0) - mean
            var += diff * diff
    var = var / n_features
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Step 3: Normalize, apply weight/bias, GELU, and transpose
    result = 0.0
    for i in range(BLOCK_SIZE):
        offs = block_start + i * n_features
        m = offs < batch_seq_idx * n_features + n_elements
        if m:
            x_val = tl.load(x_ptr + offs, mask=m, other=0.0)
            # Layer norm: (x - mean) * rstd
            normed = (x_val - mean) * rstd
            # Apply weight and bias
            out = normed * w + b
            # GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608028654 * (out + 0.04471499822557449 * out * out * out)))
            out = out * cdf
            # Store with transposed layout: output[feature, seq] for this batch
            transposed_offs = feature_idx * n_elements + (batch_seq_idx * n_elements + i)
            tl.store(output_ptr + transposed_offs, out, mask=m)


@triton.jit
def layer_norm_fused_kernel_autotuned(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    N_FEATURES: tl.constexpr,
):
    """
    Optimized LayerNorm + Transpose + GELU kernel.
    
    Each block handles one sequence position with all 512 features.
    Grid: (n_elements, 1)
    """
    # Block index for sequence dimension
    seq_idx = tl.program_id(0)
    
    # Feature offsets
    feat_offsets = tl.arange(0, N_FEATURES)
    
    # Load all features for this sequence position (contiguous access)
    x = tl.load(x_ptr + seq_idx * N_FEATURES + feat_offsets,
                mask=feat_offsets < N_FEATURES,
                other=0.0).to(tl.float32)
    
    # Load weight and bias for all features
    w = tl.load(weight_ptr + feat_offsets).to(tl.float32)
    b = tl.load(bias_ptr + feat_offsets).to(tl.float32)
    
    # LayerNorm: compute mean and variance over feature dimension (axis=0)
    mean = tl.sum(x) / N_FEATURES
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / N_FEATURES
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply affine transform
    normed = x_centered * rstd
    out = normed * w + b
    
    # GELU activation using the exact formula from PyTorch
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.04471499822557449
    inner = sqrt_2_over_pi * (out + coeff * out * out * out)
    
    # tanh using sigmoid: tanh(z) = 2 * sigmoid(2z) - 1
    exp_neg_2inner = tl.exp(-2.0 * inner)
    sigmoid_2inner = 1.0 / (1.0 + exp_neg_2inner)
    tanh_inner = 2.0 * sigmoid_2inner - 1.0
    
    # GELU = 0.5 * x * (1 + tanh)
    out_gelu = 0.5 * out * (1.0 + tanh_inner)
    
    # Store with transpose: output[feature, seq] = input[seq, feature]
    # out_gelu[feat] goes to output[feat, seq_idx]
    output_offsets = feat_offsets * n_elements + seq_idx
    tl.store(output_ptr + output_offsets, out_gelu, mask=feat_offsets < N_FEATURES)


def pattern(in_0, in_1, in_2):
    """
    Pattern: LayerNorm -> Transpose -> GELU
    This matches the computation:
        tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
        tmp_3 = tmp_2.transpose(-2, -1)
        tmp_4 = torch.nn.functional.gelu(tmp_3)
    """
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for the replacement kernel.
    in_0: bias tensor for layer norm
    in_1: weight tensor for layer norm  
    in_2: input tensor [batch, seq_len, features=512]
    """
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper function to launch the fused LayerNorm + Transpose + GELU kernel.
    
    Input: in_2 of shape [batch, seq_len, n_features] = [1, 3999, 512]
    Output: shape [batch, n_features, seq_len] = [1, 512, 3999]
    """
    batch_size, seq_len, n_features = in_2.shape
    
    # Output tensor with transposed shape
    output = torch.empty((batch_size, n_features, seq_len), 
                         dtype=in_2.dtype, 
                         device=in_2.device)
    
    # Launch kernel with one block per sequence element
    # Each block processes all 512 features for one sequence position
    layer_norm_fused_kernel_autotuned[(seq_len,)](
        in_2, in_1, in_0, output,
        seq_len,
        eps=1e-05,
        N_FEATURES=512,
    )
    
    return output


def replacement_func():
    """
    Returns the replacement function for the fused kernel.
    """
    return fused_kernel_wrapper