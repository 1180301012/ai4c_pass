import torch
import triton
import triton.language as tl


@triton.jit
def fused_gelu_add_layernorm_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr,
    out_pre_ptr, out_post_ptr,
    C: tl.constexpr, W, H_times_W, eps,
    DTYPE: tl.constexpr,
):
    """Fused kernel that computes:
    1. GELU(in_2) reshaped and transposed + in_3 -> out_pre (tmp_10)
    2. LayerNorm(out_pre) reshaped -> out_post (tmp_12)
    
    Each program processes one row (one spatial position, all channels).
    """
    pid = tl.program_id(0)
    row = pid  # row = h * W + w, ranging from 0 to H*W-1
    
    col_offsets = tl.arange(0, C)
    
    # Compute spatial coordinates from row index
    h = row // W
    w = row % W
    
    # Phase 1: GELU + reshape + add
    # Read in_2[0, col, h, w] for all channels (strided access)
    # in_2 is [1, C, H, W], so offset = col * H * W + h * W + w
    in_2_offsets = col_offsets * H_times_W + h * W + w
    in_2_vals = tl.load(in_2_ptr + in_2_offsets).to(tl.float32)
    
    # GELU exact formula: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2_inv = 0.7071067811865476  # 1/sqrt(2)
    gelu_vals = in_2_vals * 0.5 * (1.0 + tl.math.erf(in_2_vals * sqrt2_inv))
    
    # Read in_3[0, row, col] for all channels (contiguous access)
    # in_3 is [1, H*W, C], so offset = row * C + col
    in_3_vals = tl.load(in_3_ptr + row * C + col_offsets).to(tl.float32)
    
    # Add: row_vals = in_3 + gelu(in_2_reshaped)
    row_vals = in_3_vals + gelu_vals
    
    # Store pre-normalization output (tmp_10) [1, H*W, C]
    tl.store(out_pre_ptr + row * C + col_offsets, row_vals.to(DTYPE))
    
    # Phase 2: Layer normalization (in float32 for precision)
    # Compute mean over C elements
    mean = tl.sum(row_vals, axis=0) / C
    
    # Compute variance
    centered = row_vals - mean
    var = tl.sum(centered * centered, axis=0) / C
    
    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = centered * rstd
    
    # Load weight [C] and bias [C]
    w_vals = tl.load(weight_ptr + col_offsets).to(tl.float32)
    b_vals = tl.load(bias_ptr + col_offsets).to(tl.float32)
    
    # Affine transformation: weight * normed + bias
    result = w_vals * normed + b_vals
    
    # Store post-normalization output (tmp_12) [1, H, W, C]
    # Since tmp_12 is contiguous [1, H, W, C], offset = row * C + col = (h*W+w)*C + c
    tl.store(out_post_ptr + row * C + col_offsets, result.to(DTYPE))


# Dtype mapping from torch to Triton types
_DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@torch.fx.wrap
def dispatch_fused_gelu_add_layernorm(in_0, in_1, in_2, in_3, route):
    """Dispatch wrapper that routes to the appropriate kernel configuration
    based on the route string. All pass files share this same replacement_func."""
    if route == "C128_H16_W12":
        C, H, W = 128, 16, 12
    elif route == "C32_H64_W48":
        C, H, W = 32, 64, 48
    elif route == "C256_H8_W6":
        C, H, W = 256, 8, 6
    else:
        raise ValueError(f"Unknown route: {route}")
    
    HW = H * W
    eps = 1e-06
    DTYPE = _DTYPE_MAP[in_2.dtype]
    
    # Allocate output tensors
    tmp_10 = torch.empty((1, HW, C), dtype=in_2.dtype, device=in_2.device)
    tmp_12 = torch.empty((1, H, W, C), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel with one program per row (spatial position)
    grid = (HW,)
    fused_gelu_add_layernorm_kernel[grid](
        in_2, in_3, in_1, in_0,
        tmp_10, tmp_12,
        C=C, W=W, H_times_W=HW, eps=eps,
        DTYPE=DTYPE,
    )
    
    return (tmp_10, tmp_12)