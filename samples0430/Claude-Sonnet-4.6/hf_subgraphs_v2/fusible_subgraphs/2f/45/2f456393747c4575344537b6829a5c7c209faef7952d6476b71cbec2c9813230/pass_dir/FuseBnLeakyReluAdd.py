import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias, residual):
    """
    Match: batch_norm(inference) -> leaky_relu -> add(residual)
    This mirrors the exact operations in model.py after conv2d.
    """
    bn = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    out = torch.nn.functional.leaky_relu(bn, 0.01, True)
    result = out + residual
    return result


def replacement_args(x, running_mean, running_var, weight, bias, residual):
    return (x, running_mean, running_var, weight, bias, residual)


@triton.jit
def _bn_leaky_relu_add_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    res_ptr,
    out_ptr,
    HW,
    C,
    NC,
    BLOCK_HW: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # 2D grid: dim0 = spatial tile index, dim1 = N*C index
    hw_bid = tl.program_id(0)
    nc_id  = tl.program_id(1)

    # Channel index for this NC slice — no per-element integer division needed
    c_id = nc_id % C

    # Load BN params once per program as scalars (broadcast to all threads in warp)
    mean = tl.load(mean_ptr   + c_id).to(tl.float32)
    var  = tl.load(var_ptr    + c_id).to(tl.float32)
    w    = tl.load(weight_ptr + c_id).to(tl.float32)
    b    = tl.load(bias_ptr   + c_id).to(tl.float32)

    # Pre-fuse BN into a single fused-multiply-add: y = x * scale + shift
    scale = w / tl.sqrt(var + 1e-5)
    shift = b - mean * scale

    # Spatial offsets for this tile
    hw_offsets = hw_bid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    # Global linear offsets in NCHW layout
    global_offsets = nc_id * HW + hw_offsets

    # Load input activations and residual branch
    x   = tl.load(x_ptr  + global_offsets, mask=hw_mask, other=0.0)
    res = tl.load(res_ptr + global_offsets, mask=hw_mask, other=0.0)

    x_f32   = x.to(tl.float32)
    res_f32 = res.to(tl.float32)

    # Fused: BN + leaky_relu + residual add — all in float32
    y = x_f32 * scale + shift
    y = tl.where(y >= 0.0, y, 0.01 * y)
    y = y + res_f32

    # Downcast back to input dtype and store
    if IS_FP16:
        tl.store(out_ptr + global_offsets, y.to(tl.float16), mask=hw_mask)
    elif IS_BF16:
        tl.store(out_ptr + global_offsets, y.to(tl.bfloat16), mask=hw_mask)
    else:
        tl.store(out_ptr + global_offsets, y, mask=hw_mask)


def _launch_kernel(x, mean_d, var_d, weight_d, bias_d, residual, out,
                   HW, C, NC, IS_FP16, IS_BF16):
    """Static BLOCK_HW dispatch — no autotune lambda overhead per call."""
    if HW >= 4096:
        # HW=4096 (64×64): 1 tile covers a full channel, perfect alignment
        grid = (1, NC)
        _bn_leaky_relu_add_kernel[grid](
            x, mean_d, var_d, weight_d, bias_d, residual, out,
            HW, C, NC,
            BLOCK_HW=4096,
            IS_FP16=IS_FP16, IS_BF16=IS_BF16,
            num_warps=8,
        )
    elif HW >= 2048:
        # HW=3136 (56×56): 2 tiles per channel
        n_hw = (HW + 2047) // 2048
        grid = (n_hw, NC)
        _bn_leaky_relu_add_kernel[grid](
            x, mean_d, var_d, weight_d, bias_d, residual, out,
            HW, C, NC,
            BLOCK_HW=2048,
            IS_FP16=IS_FP16, IS_BF16=IS_BF16,
            num_warps=8,
        )
    else:
        # Fallback: 1024-element tiles
        n_hw = (HW + 1023) // 1024
        grid = (n_hw, NC)
        _bn_leaky_relu_add_kernel[grid](
            x, mean_d, var_d, weight_d, bias_d, residual, out,
            HW, C, NC,
            BLOCK_HW=1024,
            IS_FP16=IS_FP16, IS_BF16=IS_BF16,
            num_warps=4,
        )


@torch.fx.wrap
def bn_leaky_relu_add(x, running_mean, running_var, weight, bias, residual):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    out = torch.empty_like(x)

    device = x.device
    running_mean_d = running_mean.to(device)
    running_var_d  = running_var.to(device)
    weight_d       = weight.to(device)
    bias_d         = bias.to(device)

    IS_FP16 = (x.dtype == torch.float16)
    IS_BF16 = (x.dtype == torch.bfloat16)

    _launch_kernel(x, running_mean_d, running_var_d, weight_d, bias_d,
                   residual, out, HW, C, NC, IS_FP16, IS_BF16)

    return out


def replacement_func():
    return bn_leaky_relu_add