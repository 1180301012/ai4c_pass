import torch
import triton
import triton.language as tl


# ─── Pattern ────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 256, 16, 16)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ─── Triton kernel ───────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['HW'],
)
@triton.jit
def _fused_bn1_256_16_16_silu_kernel(
    inp_ptr,     # input  [C, HW]  (in_4 viewed as [C, HW], always contiguous)
    mean_ptr,    # running_mean [C]
    var_ptr,     # running_var  [C]
    weight_ptr,  # BN weight    [C]
    bias_ptr,    # BN bias      [C]
    out_ptr,     # output [C, HW]
    HW,          # H*W (runtime scalar)
    eps,         # BN epsilon (runtime scalar)
    BLOCK_SIZE: tl.constexpr,
):
    # One program per channel
    c = tl.program_id(0)

    # Load BN params for this channel (compute in fp32)
    mean   = tl.load(mean_ptr   + c).to(tl.float32)
    var    = tl.load(var_ptr    + c).to(tl.float32)
    w      = tl.load(weight_ptr + c).to(tl.float32)
    b      = tl.load(bias_ptr   + c).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Base offset for this channel in the [C, HW] layout
    base = c * HW

    # Process all HW spatial elements in one block
    for start in tl.range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW

        x = tl.load(inp_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        # Batch-norm inference: y = (x - mean) / sqrt(var + eps) * weight + bias
        y = (x - mean) * inv_std * w + b
        # SiLU: y * sigmoid(y)
        out = y * tl.sigmoid(y)
        tl.store(out_ptr + base + offs, out, mask=mask)


# ─── Kernel wrapper (must be decorated with @torch.fx.wrap) ─────────────────
@torch.fx.wrap
def _fuse_reshape_bn1_256_16_16_silu(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : running_mean [256]
    in_1 : running_var  [256]
    in_2 : bias         [256]
    in_3 : weight       [256]
    in_4 : input        [4, 64, 256]  (contiguous, treated as [256, 256])
    """
    C  = 256
    H  = 16
    W  = 16
    HW = H * W   # 256

    device = in_4.device
    dtype  = in_4.dtype

    # Move BN parameters to the same device/dtype as the activation tensor
    mean   = in_0.to(device=device, dtype=dtype)
    var    = in_1.to(device=device, dtype=dtype)
    weight = in_3.to(device=device, dtype=dtype)
    bias   = in_2.to(device=device, dtype=dtype)

    # View in_4 as [C, HW] (same underlying memory, strides unchanged)
    inp = in_4.view(C, HW)
    out = torch.empty_like(inp)

    _fused_bn1_256_16_16_silu_kernel[(C,)](
        inp, mean, var, weight, bias, out,
        HW=HW,
        eps=1e-05,
    )

    return out.view(1, C, H, W)


# ─── Replacement factory ─────────────────────────────────────────────────────
def replacement_func():
    return _fuse_reshape_bn1_256_16_16_silu