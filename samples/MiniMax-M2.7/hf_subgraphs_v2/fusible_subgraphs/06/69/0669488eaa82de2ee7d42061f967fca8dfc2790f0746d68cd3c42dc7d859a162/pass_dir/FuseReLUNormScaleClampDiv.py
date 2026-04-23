import torch
import triton
import triton.language as tl

# Autotune configurations for different feature sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 48}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 96}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
    ],
    key=['n_features'],
)
@triton.jit
def fused_norm_kernel_0_144(
    x_ptr,
    scalar_weight,
    out_ptr,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: relu -> flatten -> norm -> scale(0.14433756729740643) -> clamp -> div -> mul
    
    Optimized version using loop for norm computation.
    scale_factor = 0.14433756729740643
    """
    # Calculate which element this program handles
    pid = tl.program_id(0)
    
    # Each program handles one output element's feature vector
    # Compute base offset for this element's features
    base_offset = pid * n_features
    
    # Compute L2 norm using loop
    sum_squares = 0.0
    for i in range(BLOCK_SIZE):
        feat_idx = i
        if feat_idx < n_features:
            offset = base_offset + feat_idx
            val = tl.load(x_ptr + offset).to(tl.float32)
            # Apply ReLU
            val = tl.where(val > 0, val, 0.0)
            sum_squares += val * val
    
    norm = tl.sqrt(sum_squares + 1e-12)
    scaled_norm = norm * 0.14433756729740643
    # Clamp minimum
    scaled_norm = tl.where(scaled_norm > 1e-05, scaled_norm, 1e-05)
    
    # Load scalar weight (it's a scalar but we load as float)
    weight = scalar_weight.to(tl.float32)
    
    # Now compute output by dividing each feature by scaled_norm and multiplying by weight
    for i in range(BLOCK_SIZE):
        feat_idx = i
        if feat_idx < n_features:
            offset = base_offset + feat_idx
            val = tl.load(x_ptr + offset).to(tl.float32)
            # Apply ReLU
            val = tl.where(val > 0, val, 0.0)
            # Normalize and scale
            out_val = (val / scaled_norm) * weight
            tl.store(out_ptr + offset, out_val, mask=True)


@torch.fx.wrap
def fused_norm_wrapper_0_144(x, scalar_weight):
    """
    Wrapper function to launch the fused normalization kernel with scale 0.14433756729740643.
    """
    batch, channels, H, W = x.shape
    n_features = H * W
    
    # Output shape matches input
    out = torch.empty_like(x)
    
    # Launch kernel with one program per output element (batch * channels)
    grid = (batch * channels,)
    
    fused_norm_kernel_0_144[grid](
        x,
        scalar_weight,
        out,
        n_features,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the pattern: relu -> flatten -> norm -> scale -> clamp -> div -> mul
    with scale_factor = 0.14433756729740643
    """
    relu_out = torch.nn.functional.relu(in_1, inplace = True)
    flat_out = torch.flatten(relu_out, 2)
    norm_out = torch.functional.norm(flat_out, dim = -1, keepdim = True)
    scale_out = norm_out * 0.14433756729740643
    clamp_out = scale_out.clamp(min = 1e-05)
    div_out = flat_out / clamp_out
    result = div_out * in_0
    return result


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement function.
    """
    # Pass scalar weight tensor directly (shape [1])
    return (in_1, in_0)


def replacement_func():
    """
    Returns the replacement function.
    """
    return fused_norm_wrapper_0_144