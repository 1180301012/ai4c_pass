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
def fused_norm_kernel_0_072_float32(
    x_ptr,
    scalar_weight,
    out_ptr,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: relu -> flatten -> norm -> scale(0.07216878364870322) -> clamp -> div -> mul
    For float32 models that use tmp_0 = in_0 pattern.
    """
    # Calculate which element this program handles
    pid = tl.program_id(0)
    
    # Each program handles one output element's feature vector
    base_offset = pid * n_features
    
    # Compute L2 norm using loop
    sum_squares = 0.0
    for i in range(BLOCK_SIZE):
        feat_idx = i
        if feat_idx < n_features:
            offset = base_offset + feat_idx
            val = tl.load(x_ptr + offset).to(tl.float32)
            val = tl.where(val > 0, val, 0.0)
            sum_squares += val * val
    
    norm = tl.sqrt(sum_squares + 1e-12)
    scaled_norm = norm * 0.07216878364870322
    scaled_norm = tl.where(scaled_norm > 1e-05, scaled_norm, 1e-05)
    
    weight = scalar_weight.to(tl.float32)
    
    for i in range(BLOCK_SIZE):
        feat_idx = i
        if feat_idx < n_features:
            offset = base_offset + feat_idx
            val = tl.load(x_ptr + offset).to(tl.float32)
            val = tl.where(val > 0, val, 0.0)
            out_val = (val / scaled_norm) * weight
            tl.store(out_ptr + offset, out_val, mask=True)


@torch.fx.wrap
def fused_norm_wrapper_0_072_float32(x, scalar_weight):
    """Wrapper for float32 models with tmp_0 = in_0 pattern."""
    batch, channels, H, W = x.shape
    n_features = H * W
    out = torch.empty_like(x)
    grid = (batch * channels,)
    
    fused_norm_kernel_0_072_float32[grid](
        x,
        scalar_weight,
        out,
        n_features,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the float32 pattern with tmp_0 = in_0:
    relu -> flatten -> norm -> scale -> clamp -> div -> mul
    scale_factor = 0.07216878364870322
    """
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_1 = None
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_3 = None
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_4 = None
    tmp_6 = tmp_2 / tmp_5
    tmp_2 = tmp_5 = None
    tmp_7 = tmp_6 * tmp_0
    tmp_6 = tmp_0 = None
    return tmp_7


def replacement_args(in_0, in_1):
    """Extract arguments for the replacement function."""
    return (in_1, in_0)


def replacement_func():
    """Returns the replacement function."""
    return fused_norm_wrapper_0_072_float32