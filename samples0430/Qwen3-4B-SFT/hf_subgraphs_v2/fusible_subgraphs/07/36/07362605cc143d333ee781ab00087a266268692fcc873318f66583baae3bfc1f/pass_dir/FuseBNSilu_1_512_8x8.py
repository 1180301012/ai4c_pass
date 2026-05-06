"""
Fuses batch_norm (inference) + SiLU activation into a single Triton kernel.
Matches the pattern: reshape(1, 512, 8, 8) -> batch_norm -> silu
Input shape: (4, 128, 64) -> viewed as (1, 512, 8, 8), 32768 elements.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: one program per BLOCK_SIZE elements (flat index)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _bn_silu_kernel_512(
    in_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,          # total elements = 32768
    C: tl.constexpr,        # 512
    HW: tl.constexpr,       # 64  (8*8)
    EPS: tl.constexpr,      # 1e-5
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Flat -> channel-major decomposition
    # [1, C, H, W]: element at flat idx has channel = idx // HW, spatial = idx % HW
    channel_ids = offsets // HW          # [0, 511] for any valid element

    # Load per-channel BN statistics (gather of scalars – cache-friendly since
    # there are only 512 distinct values but many repeats per block)
    mean  = tl.load(mean_ptr  + channel_ids, mask=mask, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr   + channel_ids, mask=mask, other=1.0).to(tl.float32)
    w     = tl.load(weight_ptr + channel_ids, mask=mask, other=1.0).to(tl.float32)
    b     = tl.load(bias_ptr  + channel_ids, mask=mask, other=0.0).to(tl.float32)

    # Load input (bfloat16 / float16)
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # BN inference: out = (x - mean) / sqrt(var + eps) * weight + bias
    x_f32 = x.to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + EPS)
    y = (x_f32 - mean) * inv_std * w + b

    # SiLU: y * sigmoid(y)
    out = y * tl.sigmoid(y)

    # Store
    tl.store(out_ptr + offsets, out.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_bn_silu_512_8x8(in_4, in_0, in_1, in_3, in_2):
    """
    in_4 : input tensor  (4, 128, 64)  on CUDA
    in_0 : running_mean  (512,)         on CPU
    in_1 : running_var   (512,)         on CPU
    in_3 : weight (gamma) (512,)        on CPU
    in_2 : bias   (beta)  (512,)        on CPU
    Returns tensor of shape (1, 512, 8, 8) on CUDA with same dtype as in_4.
    """
    C    = 512
    HW   = 64          # 8 * 8
    N    = 32768       # 1 * 512 * 64

    # GPU pointers for all per-channel parameters
    mean_gpu  = in_0.to(in_4.device)
    var_gpu   = in_1.to(in_4.device)
    w_gpu     = in_3.to(in_4.device)
    b_gpu     = in_2.to(in_4.device)

    out = torch.empty(1, C, 8, 8, dtype=in_4.dtype, device=in_4.device)

    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _bn_silu_kernel_512[grid](
        in_4, mean_gpu, var_gpu, w_gpu, b_gpu, out,
        N=N, C=C, HW=HW, EPS=1e-5,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------
def pattern(in_4, in_0, in_1, in_3, in_2):
    tmp_4 = in_4.reshape(1, 512, 8, 8)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)


def replacement_func():
    return fused_bn_silu_512_8x8