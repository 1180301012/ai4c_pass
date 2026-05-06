import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match BN-inference + LeakyReLU + residual-add
# ---------------------------------------------------------------------------
def pattern(x, running_mean, running_var, weight, bias, residual):
    bn     = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    lr     = torch.nn.functional.leaky_relu(bn, 0.01, True)
    result = lr + residual
    return result


def replacement_args(x, running_mean, running_var, weight, bias, residual):
    return (x, running_mean, running_var, weight, bias, residual)


# ---------------------------------------------------------------------------
# Triton kernel: fused BN (inference) + LeakyReLU + residual add
#
# Tensor layout assumed: contiguous NCHW  [N, C, H, W]
# Grid: (N*C, ceil(H*W / BLOCK_SIZE))
#   pid_nc selects the (batch, channel) slice — channel c = pid_nc % C
#   pid_hw selects the tile within the spatial HW plane
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["HW", "C", "N"],
)
@triton.jit
def _bn_lrelu_add_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    C,
    HW,
    eps: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)   # ∈ [0, N*C)
    pid_hw = tl.program_id(1)   # tiles HW dimension

    c = pid_nc % C
    n = pid_nc // C

    # Load per-channel BN parameters (4 scalars, broadcast to all threads)
    mean   = tl.load(mean_ptr   + c).to(tl.float32)
    var    = tl.load(var_ptr    + c).to(tl.float32)
    w      = tl.load(weight_ptr + c).to(tl.float32)
    b      = tl.load(bias_ptr   + c).to(tl.float32)
    scale  = w / tl.sqrt(var + eps)
    shift  = b - mean * scale

    hw_start = pid_hw * BLOCK_SIZE
    hw_offs  = hw_start + tl.arange(0, BLOCK_SIZE)
    mask     = hw_offs < HW
    base     = (n * C + c) * HW

    x   = tl.load(x_ptr     + base + hw_offs, mask=mask, other=0.0).to(tl.float32)
    rem = tl.load(residual_ptr + base + hw_offs, mask=mask, other=0.0).to(tl.float32)

    y = x * scale + shift
    y = tl.where(y >= 0.0, y, y * 0.01)
    y = y + rem

    if IS_FP16:
        tl.store(out_ptr + base + hw_offs, y.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + base + hw_offs, y.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + base + hw_offs, y.to(tl.float32), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper called by the pattern-replacement framework
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_bn_lrelu_add(x, running_mean, running_var, weight, bias, residual):
    N  = x.shape[0]
    C  = x.shape[1]
    H  = x.shape[2]
    W  = x.shape[3]
    HW = H * W

    IS_FP16 = x.dtype == torch.float16
    IS_BF16 = x.dtype == torch.bfloat16

    out  = torch.empty_like(x)
    NC   = N * C

    grid = lambda meta: (NC, triton.cdiv(HW, meta["BLOCK_SIZE"]))

    _bn_lrelu_add_kernel[grid](
        x, running_mean, running_var, weight, bias,
        residual, out,
        C, HW,
        eps=1e-5,
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
    )

    return out


def replacement_func():
    return fused_bn_lrelu_add