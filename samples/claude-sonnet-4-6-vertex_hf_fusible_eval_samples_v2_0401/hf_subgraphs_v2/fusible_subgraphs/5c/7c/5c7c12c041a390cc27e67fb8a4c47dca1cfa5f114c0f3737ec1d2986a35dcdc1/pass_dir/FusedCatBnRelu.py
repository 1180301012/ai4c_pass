import torch
import triton
import triton.language as tl


def pattern(a, b, running_mean, running_var, weight, bias):
    """
    Match: cat([a, b], dim=1) -> batch_norm (inference) -> relu
    This avoids materializing the intermediate concatenated tensor.
    """
    cat_out = torch.cat([a, b], 1)
    bn_out = torch.nn.functional.batch_norm(cat_out, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    out = torch.nn.functional.relu(bn_out, inplace=False)
    return out


def replacement_args(a, b, running_mean, running_var, weight, bias):
    return (a, b, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['N', 'C1', 'C2', 'HW'],
)
@triton.jit
def fused_cat_bn_relu_kernel(
    a_ptr, b_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C1, C2, HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused kernel: concatenate two NCHW tensors along C, apply BN (inference), apply ReLU.
    Grid: (N * (C1+C2), ceil(HW / BLOCK_HW))
    """
    nc_pid = tl.program_id(0)
    hw_pid = tl.program_id(1)

    C = C1 + C2
    n_idx = nc_pid // C
    c_idx = nc_pid % C

    # Load BN parameters for this channel (scalar loads)
    mean_val = tl.load(mean_ptr + c_idx).to(tl.float32)
    var_val  = tl.load(var_ptr  + c_idx).to(tl.float32)
    w_val    = tl.load(weight_ptr + c_idx).to(tl.float32)
    b_val    = tl.load(bias_ptr + c_idx).to(tl.float32)

    # Precompute BN scale and shift
    inv_std = w_val / tl.sqrt(var_val + 1e-3)
    shift   = b_val - mean_val * inv_std

    # Spatial tile
    hw_start = hw_pid * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask  = hw_offs < HW

    # Determine if channel comes from tensor a or tensor b
    is_a = c_idx < C1

    # Safe indices: clamp to avoid out-of-bounds address computation
    # (masked loads ensure we never actually access invalid memory)
    a_c = tl.where(is_a, c_idx, C1 - 1)          # clamped a channel
    b_c = tl.where(is_a, 0, c_idx - C1)           # clamped b channel

    a_offset = (n_idx * C1 + a_c) * HW + hw_offs
    b_offset = (n_idx * C2 + b_c) * HW + hw_offs

    a_loaded = tl.load(a_ptr + a_offset, mask=hw_mask & is_a,  other=0.0).to(tl.float32)
    b_loaded = tl.load(b_ptr + b_offset, mask=hw_mask & ~is_a, other=0.0).to(tl.float32)

    x = tl.where(is_a, a_loaded, b_loaded)

    # BN (inference mode) + ReLU
    out_val = tl.maximum(x * inv_std + shift, 0.0)

    out_offset = (n_idx * C + c_idx) * HW + hw_offs
    tl.store(out_ptr + out_offset, out_val, mask=hw_mask)


@torch.fx.wrap
def fused_cat_bn_relu(a, b, running_mean, running_var, weight, bias):
    """
    Wrapper: fused cat + BN (inference) + ReLU without intermediate cat tensor.
    """
    N  = a.shape[0]
    C1 = a.shape[1]
    C2 = b.shape[1]
    H  = a.shape[2]
    W  = a.shape[3]
    C  = C1 + C2
    HW = H * W

    out = torch.empty((N, C, H, W), dtype=a.dtype, device=a.device)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_HW']))

    fused_cat_bn_relu_kernel[grid](
        a, b,
        running_mean, running_var, weight, bias,
        out,
        N, C1, C2, HW,
    )

    return out


def replacement_func():
    return fused_cat_bn_relu