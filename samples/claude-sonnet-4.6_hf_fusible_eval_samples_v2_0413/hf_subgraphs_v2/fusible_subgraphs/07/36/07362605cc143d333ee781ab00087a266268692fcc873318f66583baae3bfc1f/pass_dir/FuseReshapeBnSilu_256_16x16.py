import torch
import triton
import triton.language as tl


# Fused kernel: batch_norm (inference) + SiLU for C=256, HW=256 (16x16)
# Each program handles one channel (256 spatial elements)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['C'],
)
@triton.jit
def fused_bn_silu_256_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C,
    eps: tl.constexpr,
):
    # One program per channel; each processes HW=256 elements
    c_id = tl.program_id(0)

    # Load per-channel BN parameters (scalar loads, promote to fp32)
    mean_val  = tl.load(mean_ptr  + c_id).to(tl.float32)
    var_val   = tl.load(var_ptr   + c_id).to(tl.float32)
    w_val     = tl.load(weight_ptr + c_id).to(tl.float32)
    b_val     = tl.load(bias_ptr  + c_id).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var_val + eps)
    # Fuse scale and shift: y = x*scale + shift
    scale = inv_std * w_val
    shift = b_val - mean_val * scale

    # Pointer base for this channel's 256-element block
    base    = c_id * 256
    offsets = tl.arange(0, 256)

    # Load input (fp16/bf16 → fp32), apply BN, then SiLU
    x = tl.load(input_ptr + base + offsets).to(tl.float32)
    x = x * scale + shift
    # SiLU: x * sigmoid(x)
    x = x * tl.sigmoid(x)

    # Store back (fp32 → original dtype via pointer type)
    tl.store(output_ptr + base + offsets, x)


@torch.fx.wrap
def triton_fused_bn_silu_256(mean, var, bias, weight, x):
    """
    Inputs:
      mean, var : running stats  [256], CPU tensors → moved to GPU
      bias      : BN bias        [256], CPU
      weight    : BN weight      [256], CPU
      x         : activations    [4, 64, 256] on GPU
    Output:
      [1, 256, 16, 16]
    """
    C  = 256
    HW = 256  # 16 * 16

    # Flatten input to [C, HW]; this is a zero-copy view
    x_flat = x.reshape(C, HW)

    # Move BN parameters to GPU (they live on CPU in model)
    mean_d   = mean.to(x.device)
    var_d    = var.to(x.device)
    weight_d = weight.to(x.device)
    bias_d   = bias.to(x.device)

    # Allocate output with same dtype/device as input
    out = torch.empty_like(x_flat)

    # Launch: one program per channel
    fused_bn_silu_256_kernel[(C,)](
        x_flat,
        mean_d,
        var_d,
        weight_d,
        bias_d,
        out,
        C,
        eps=1e-5,
    )

    # Reshape to expected output shape
    return out.reshape(1, C, 16, 16)


# ---------------------------------------------------------------------------
# Pattern: reshape(1,256,16,16) → batch_norm → silu
# Must mirror model.py exactly (positional args, inplace keyword)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 256, 16, 16)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    # Order: (mean, var, bias, weight, x)
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return triton_fused_bn_silu_256