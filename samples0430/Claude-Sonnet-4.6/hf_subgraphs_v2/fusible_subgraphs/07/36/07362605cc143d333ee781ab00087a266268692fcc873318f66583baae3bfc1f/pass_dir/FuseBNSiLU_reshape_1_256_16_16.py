"""
Fused pass: reshape(1,256,16,16) + batch_norm (inference) + silu
Handles both float16 and bfloat16 inputs.

Input shapes:
  in_0: running_mean [256]   (CPU)
  in_1: running_var  [256]   (CPU)
  in_2: bias         [256]   (CPU)
  in_3: weight       [256]   (CPU)
  in_4: input        [4,64,256] (CUDA)

Output: (Tensor[1,256,16,16],)
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: one block per channel, BLOCK_SIZE == S (aligned, no mask)
# ---------------------------------------------------------------------------
@triton.jit
def _bn_silu_kernel_256(
    x_ptr,      # [C*S] flattened input
    mean_ptr,   # [C]   running mean
    var_ptr,    # [C]   running var
    w_ptr,      # [C]   weight (gamma)
    b_ptr,      # [C]   bias   (beta)
    out_ptr,    # [C*S] output
    eps,        # BN epsilon (1e-5)
    S: tl.constexpr,   # spatial elements per channel = 256
):
    """Grid: (C,).  Block c handles spatial range [c*S, c*S+S)."""
    c = tl.program_id(0)
    s_idx = tl.arange(0, S)

    # Load BN params for this channel (scalar → broadcast)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    var  = tl.load(var_ptr  + c).to(tl.float32)
    w    = tl.load(w_ptr    + c).to(tl.float32)
    b    = tl.load(b_ptr    + c).to(tl.float32)

    # Load input elements for this channel
    x    = tl.load(x_ptr + c * S + s_idx)
    x_f  = x.to(tl.float32)

    # Inference BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (x_f - mean) * inv_std * w + b

    # SiLU: y * sigmoid(y)
    out = y * tl.sigmoid(y)

    # Store in original dtype
    tl.store(out_ptr + c * S + s_idx, out.to(x.dtype))


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 256, 16, 16)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_bn_silu_256(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 = running_mean [256] (CPU bfloat16/float16)
    in_1 = running_var  [256] (CPU)
    in_2 = bias         [256] (CPU)
    in_3 = weight       [256] (CPU)
    in_4 = input [4,64,256]   (CUDA)
    Returns: (output [1,256,16,16],)
    """
    device = in_4.device
    dtype  = in_4.dtype
    C, S   = 256, 256  # channels, spatial elements per channel (16*16)

    # Move BN params to the same device as the input
    mean   = in_0.to(device=device)
    var    = in_1.to(device=device)
    weight = in_3.to(device=device)
    bias   = in_2.to(device=device)

    # Flatten input (handles non-contiguous transposes via copy-on-reshape)
    x_flat   = in_4.reshape(C * S)
    out_flat = torch.empty(C * S, dtype=dtype, device=device)

    # Launch: one block per channel, 256 threads (8 warps)
    _bn_silu_kernel_256[(C,)](
        x_flat, mean, var, weight, bias, out_flat,
        1e-05,   # eps
        S,       # constexpr S=256
        num_warps=8,
    )

    return (out_flat.reshape(1, C, 16, 16),)


def replacement_func():
    return fused_bn_silu_256