import torch
import triton
import triton.language as tl

@triton.jit
def weighted_sum_softmax_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_elements,
    K: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TILE_K: tl.constexpr,
    dtype_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: computes out = sum_c(x[:, c, :, :, :] * softmax_w[:, c])
    where softmax_w = softmax(weight, dim=1)
    
    Uses 2D grid: grid_x = H*W (spatial positions), each program processes TILE_K K values.
    x shape: [1, 2, 256, H, W]
    weight shape: [1, 2, 256, 1, 1]
    """
    pid = tl.program_id(0)
    
    # Compute h, w from flat index (each program handles one spatial position)
    h_idx = pid // W
    w_idx = pid % W
    
    # Compute base offset for x data at this (h, w)
    x_base = h_idx * W + w_idx
    
    # Process TILE_K K values in this program
    k_start = tl.program_id(1) * TILE_K
    k_range = tl.arange(0, TILE_K)
    
    # Load weight values for softmax (TILE_K values)
    w0 = tl.load(weight_ptr + k_start + k_range)
    w1 = tl.load(weight_ptr + 256 + k_start + k_range)
    
    # Cast to float32 for exp
    w0_f32 = w0.to(tl.float32)
    w1_f32 = w1.to(tl.float32)
    
    # Softmax computation for TILE_K values
    exp_w0 = tl.exp(w0_f32)
    exp_w1 = tl.exp(w1_f32)
    sum_exp = exp_w0 + exp_w1 + 1e-10
    softmax_w0 = exp_w0 / sum_exp
    softmax_w1 = exp_w1 / sum_exp
    
    # Load x values for TILE_K K values at this (h, w) position
    # x[c, k, h, w] -> flat: c*256*H*W + k*H*W + h*W + w
    x0 = tl.load(x_ptr + (k_start + k_range) * H * W + x_base).to(tl.float32)
    x1 = tl.load(x_ptr + 256 * H * W + (k_start + k_range) * H * W + x_base).to(tl.float32)
    
    # Compute weighted sum and convert to output dtype
    result = (x0 * softmax_w0 + x1 * softmax_w1).to(dtype_val)
    
    # Store result
    # out[k, h, w] -> flat: k*H*W + h*W + w
    out_offsets = (k_start + k_range) * H * W + x_base
    
    # Mask for valid K values (within bounds)
    k_mask = (k_start + k_range) < K
    tl.store(out_ptr + out_offsets, result, mask=k_mask)


@torch.fx.wrap
def weighted_sum_softmax(x, weight):
    """
    Fused kernel for: softmax(weight, dim=1) * x followed by sum(dim=1)
    
    Args:
        x: Input tensor of shape [1, 2, 256, H, W]
        weight: Weight tensor of shape [1, 2, 256, 1, 1]
    
    Returns:
        Output tensor of shape [1, 256, H, W]
    """
    B, C, K, H, W = x.shape  # B=1, C=2, K=256
    
    # Tile size for processing multiple K values per program
    TILE_K = 64
    
    # Map torch dtype to Triton dtype
    x_dtype = x.dtype
    if x_dtype == torch.float32:
        triton_dtype = tl.float32
    elif x_dtype == torch.float16:
        triton_dtype = tl.float16
    elif x_dtype == torch.bfloat16:
        triton_dtype = tl.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {x_dtype}")
    
    # Output shape [1, K, H, W]
    out = torch.empty((1, K, H, W), dtype=x.dtype, device=x.device)
    
    # 2D grid: (H*W) x (K/TILE_K)
    grid_x = H * W  # spatial positions
    grid_y = (K + TILE_K - 1) // TILE_K  # K blocks
    
    BLOCK_SIZE = 1024
    
    weighted_sum_softmax_kernel[(grid_x, grid_y)](
        x_ptr=x,
        weight_ptr=weight,
        out_ptr=out,
        n_elements=K * H * W,
        K=K,
        H=H,
        W=W,
        TILE_K=TILE_K,
        dtype_val=triton_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """Match the pattern: softmax -> multiply -> sum"""
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return weighted_sum_softmax