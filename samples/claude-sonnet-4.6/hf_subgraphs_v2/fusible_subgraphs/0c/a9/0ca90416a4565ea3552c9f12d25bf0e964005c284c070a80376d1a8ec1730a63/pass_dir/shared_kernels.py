import torch
import triton
import triton.language as tl


# ── Triton kernel: inference-mode BatchNorm ───────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
    ],
    key=['spatial'],
)
@triton.jit
def _bn_inference_kernel(
    x_ptr, y_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Inference-mode BatchNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    Grid: (N*C, ceil(spatial / BLOCK_SIZE))
    """
    pid_nc = tl.program_id(0)   # one program per (n, c) pair
    pid_s  = tl.program_id(1)   # tile along spatial dim

    c = pid_nc % C

    # Per-channel scalars — upcast to f32 for numerical stability
    mean   = tl.load(mean_ptr   + c).to(tl.float32)
    var    = tl.load(var_ptr    + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias   = tl.load(bias_ptr   + c).to(tl.float32)

    scale = weight / tl.sqrt(var + eps)
    shift = bias - mean * scale

    base    = pid_nc * spatial
    offsets = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < spatial

    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    y = x.to(tl.float32) * scale + shift
    tl.store(y_ptr + base + offsets, y.to(x.dtype), mask=mask)


# ── Parameter cache: avoid repeated CPU→GPU transfers for frozen BN stats ─────
# Keys are (data_ptr, numel, dtype) so different param tensors never collide.

_param_cache: dict = {}


def _ensure_cuda(param, device):
    """
    Return `param` on `device`.
    If `param` is already on `device`, return it directly (zero overhead).
    Otherwise move it to `device` and cache the result so subsequent calls
    reuse the CUDA tensor without re-allocating or re-transferring.
    """
    if param.device == device:
        return param
    key = (param.data_ptr(), param.numel(), param.dtype)
    if key not in _param_cache:
        _param_cache[key] = torch.as_tensor(param, device=device)
    return _param_cache[key]


# ── Python helper ─────────────────────────────────────────────────────────────

def _run_bn_inference(x, running_mean, running_var, bias, weight):
    """
    Dispatch inference-mode BN via Triton.
    BN parameters that live on CPU are moved to CUDA once and then cached.
    """
    device = x.device
    running_mean = _ensure_cuda(running_mean, device)
    running_var  = _ensure_cuda(running_var,  device)
    weight       = _ensure_cuda(weight,       device)
    bias         = _ensure_cuda(bias,         device)

    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    spatial = H * W
    out     = torch.empty_like(x)

    grid = lambda meta: (N * C, triton.cdiv(spatial, meta['BLOCK_SIZE']))
    _bn_inference_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        C, spatial,
        0.001,   # eps
    )
    return out


# ── FX-wrapped entry point ────────────────────────────────────────────────────
# @torch.fx.wrap makes this opaque to the FX tracer → single graph node,
# single output tensor → works cleanly with SubgraphRewriter.

@torch.fx.wrap
def bn_wrapped(in_4, in_0, in_1, in_2, in_3):
    """
    FX-opaque wrapper for Triton BN inference.
    Signature mirrors replacement_args order:
        in_4 = input,       in_0 = running_mean,
        in_1 = running_var, in_2 = bias, in_3 = weight
    """
    return _run_bn_inference(in_4, in_0, in_1, in_2, in_3)