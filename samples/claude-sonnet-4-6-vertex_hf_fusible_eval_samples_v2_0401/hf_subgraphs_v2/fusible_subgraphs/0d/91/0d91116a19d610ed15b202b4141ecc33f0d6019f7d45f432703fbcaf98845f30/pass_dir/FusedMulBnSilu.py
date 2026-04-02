import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16),
        triton.Config({'BLOCK_HW': 1024}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def fused_mul_bn_silu_kernel(
    x_ptr,
    sigmoid_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused kernel: elementwise multiply + batch norm (inference) + SiLU.

    Grid: (N*C, ceil(HW / BLOCK_HW))
    Each program handles BLOCK_HW spatial elements for one (n, c) pair.

    Computation per element:
      y = (x[n,c,h,w] * sig[n,c] - mean[c]) / sqrt(var[c] + eps) * weight[c] + bias[c]
      z = y * sigmoid(y)   (SiLU)
    """
    nc_idx = tl.program_id(0)   # flat (n,c) index
    hw_block = tl.program_id(1) # spatial block index

    c_idx = nc_idx % C

    # ---- Load per-channel BN parameters in float32 for numerical precision ----
    mean_c   = tl.load(mean_ptr   + c_idx).to(tl.float32)
    var_c    = tl.load(var_ptr    + c_idx).to(tl.float32)
    weight_c = tl.load(weight_ptr + c_idx).to(tl.float32)
    bias_c   = tl.load(bias_ptr   + c_idx).to(tl.float32)

    # Precompute BN scale and shift
    inv_std = 1.0 / tl.sqrt(var_c + 1e-5)
    scale   = weight_c * inv_std
    shift   = bias_c - mean_c * scale

    # ---- Load the sigmoid (SE) value for this (n, c) pair ----
    sig_val = tl.load(sigmoid_ptr + nc_idx).to(tl.float32)

    # Merge the sigmoid scale with the BN scale so we do one multiply per element
    combined_scale = sig_val * scale

    # ---- Process spatial block ----
    hw_start = hw_block * BLOCK_HW
    offsets  = hw_start + tl.arange(0, BLOCK_HW)
    mask     = offsets < HW

    base = nc_idx * HW

    # Load x (keep original dtype handle for dtype-agnostic store)
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)

    # Fused: mul * combined_scale + shift → SiLU
    y = x.to(tl.float32) * combined_scale + shift
    z = y * tl.sigmoid(y)

    # Store with original dtype
    tl.store(out_ptr + base + offsets, z.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_mul_bn_silu(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Replacement for: mul(in_5, in_4) → batch_norm(…, in_0, in_1, in_3, in_2) → silu

    Arguments (matching the pattern order):
      in_0 : running_mean  [C]         – CPU tensor
      in_1 : running_var   [C]         – CPU tensor
      in_2 : bias          [C]         – CPU tensor
      in_3 : weight        [C]         – CPU tensor
      in_4 : sigmoid       [N,C,1,1]   – CUDA tensor
      in_5 : x             [N,C,H,W]   – CUDA tensor
    """
    x           = in_5          # [N, C, H, W]
    sigmoid_val = in_4          # [N, C, 1, 1]

    N, C, H, W = x.shape
    HW     = H * W
    dtype  = x.dtype
    device = x.device

    # Transfer small CPU tensors to GPU with the same dtype as the main tensor
    mean_gpu   = in_0.to(device=device, dtype=dtype)
    var_gpu    = in_1.to(device=device, dtype=dtype)
    bias_gpu   = in_2.to(device=device, dtype=dtype)
    weight_gpu = in_3.to(device=device, dtype=dtype)

    # Flatten spatial dimensions for the kernel
    x_flat   = x.reshape(N * C, HW)
    sig_flat = sigmoid_val.reshape(N * C)  # [N,C,1,1] → [N*C]
    out_flat = torch.empty_like(x_flat)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_HW']))

    fused_mul_bn_silu_kernel[grid](
        x_flat, sig_flat,
        mean_gpu, var_gpu, weight_gpu, bias_gpu,
        out_flat,
        C, HW,
    )

    return out_flat.reshape(N, C, H, W)


# ---------------------------------------------------------------------------
# Pattern / replacement interface required by the AI4C framework
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Matches the subgraph:
      tmp_4 = in_5 * in_4
      tmp_5 = F.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
      tmp_6 = F.silu(tmp_5, inplace=True)
      return (tmp_6,)
    """
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_mul_bn_silu