"""
Fused pass: in-place-add + GELU + batch_norm_inference + identity(0+x)

Pattern (from model.py):
    in_4 += in_5
    tmp_5 = gelu(in_4,  approximate='none')
    tmp_6 = batch_norm(tmp_5, running_mean=in_0, running_var=in_1,
                       weight=in_3, bias=in_2, training=False, momentum=0.1, eps=1e-5)
    tmp_7 = 0 + tmp_6          # identity
    return (tmp_5, tmp_7)

Kernel design:
  - 2-D grid: (N*C,  ceil(HW / BLOCK_SIZE))
  - Each thread-block owns a fixed (n, c) slice so BN parameters (mean, var, weight, bias)
    are loaded once as scalars and broadcast across the spatial block.
  - Arithmetic is performed in fp32; results are stored back in the original dtype.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_add_gelu_bn_kernel(
    in4_ptr, in5_ptr,           # [N, C, H, W]  (input tensors)
    mean_ptr, var_ptr,          # [C]  running stats
    weight_ptr, bias_ptr,       # [C]  affine params
    gelu_out_ptr, bn_out_ptr,   # [N, C, H, W]  outputs
    HW,                         # H * W   (runtime)
    C,                          # num channels (runtime)
    eps,                        # batch-norm epsilon
    BLOCK_SIZE: tl.constexpr,
):
    # ---- which (n, c) slice and which spatial tile ----
    pid_nc = tl.program_id(0)   # flattened (n, c) index
    pid_hw = tl.program_id(1)   # spatial tile index

    c = pid_nc % C              # channel for this block

    # ---- BN parameters: single scalar load per channel ----
    mean   = tl.load(mean_ptr   + c).to(tl.float32)
    var    = tl.load(var_ptr    + c).to(tl.float32)
    w      = tl.load(weight_ptr + c).to(tl.float32)
    b      = tl.load(bias_ptr   + c).to(tl.float32)

    # pre-fuse scale/shift:  out = x * scale + shift
    scale = w / tl.sqrt(var + eps)
    shift = b - mean * scale

    # ---- spatial tile ----
    hw_start   = pid_hw * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask    = hw_offsets < HW

    global_offsets = pid_nc * HW + hw_offsets

    # ---- load inputs ----
    x4 = tl.load(in4_ptr + global_offsets, mask=hw_mask, other=0.0).to(tl.float32)
    x5 = tl.load(in5_ptr + global_offsets, mask=hw_mask, other=0.0).to(tl.float32)

    # ---- add ----
    x = x4 + x5

    # ---- exact GELU:  x * 0.5 * (1 + erf(x / sqrt(2))) ----
    gelu_x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))

    # ---- store gelu output (tmp_5) ----
    tl.store(gelu_out_ptr + global_offsets, gelu_x, mask=hw_mask)

    # ---- batch-norm (inference):  gelu_x * scale + shift ----
    bn_out = gelu_x * scale + shift

    # ---- store bn output (tmp_7 = 0 + bn_out  ≡  bn_out) ----
    tl.store(bn_out_ptr + global_offsets, bn_out, mask=hw_mask)


# ---------------------------------------------------------------------------
# Python wrapper  (must be @torch.fx.wrap so FX doesn't trace inside)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_gelu_bn(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0 : running_mean  [C]
    in_1 : running_var   [C]
    in_2 : bias          [C]
    in_3 : weight        [C]
    in_4 : first  input  [N, C, H, W]
    in_5 : second input  [N, C, H, W]

    Returns (gelu_out, bn_out)  matching the original (tmp_5, tmp_7).
    """
    N, C, H, W = in_4.shape
    HW = H * W
    NC = N * C

    gelu_out = torch.empty_like(in_4)
    bn_out   = torch.empty_like(in_4)

    # 2-D grid: dim-0 = NC slices,  dim-1 = spatial tiles
    grid = lambda meta: (NC, triton.cdiv(HW, meta['BLOCK_SIZE']))

    _fused_add_gelu_bn_kernel[grid](
        in4_ptr      = in_4,
        in5_ptr      = in_5,
        mean_ptr     = in_0,
        var_ptr      = in_1,
        weight_ptr   = in_3,
        bias_ptr     = in_2,
        gelu_out_ptr = gelu_out,
        bn_out_ptr   = bn_out,
        HW           = HW,
        C            = C,
        eps          = 1e-5,
    )

    return (gelu_out, bn_out)


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    in_4 += in_5
    tmp_5 = torch.nn.functional.gelu(in_4, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return (tmp_5, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_add_gelu_bn