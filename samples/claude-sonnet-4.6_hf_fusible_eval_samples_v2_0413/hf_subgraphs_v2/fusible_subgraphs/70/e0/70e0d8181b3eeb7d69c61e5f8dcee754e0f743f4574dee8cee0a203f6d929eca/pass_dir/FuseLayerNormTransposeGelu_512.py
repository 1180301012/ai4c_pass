import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: layer_norm -> transpose(-2,-1) -> gelu
# Shapes: in_2=[1,3999,512], in_1(weight)=[512], in_0(bias)=[512]
# Output: [1, 512, 3999]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Single fused kernel: LN + GELU + coalesced transposed write
#
#  Grid: (ceil(N/TILE_N), B)
#  Each program handles TILE_N consecutive rows:
#    • Reads [TILE_N, C] input   – coalesced along C
#    • 2-D LN reduction along axis=1 (C dim): with TILE_N=num_warps*2,
#      every warp owns exactly one row → pure warp-shuffle reduction
#    • GELU (exact erf-based)
#    • tl.trans → [C, TILE_N] + coalesced store along N
#    • No two programs share an output cache-line
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ---- TILE_N=4 ---------------------------------------------------
        triton.Config({'TILE_N': 4,  'BLOCK_C': 512}, num_warps=4,  num_stages=2),
        triton.Config({'TILE_N': 4,  'BLOCK_C': 512}, num_warps=8,  num_stages=2),
        triton.Config({'TILE_N': 4,  'BLOCK_C': 512}, num_warps=4,  num_stages=3),
        triton.Config({'TILE_N': 4,  'BLOCK_C': 512}, num_warps=8,  num_stages=3),
        # ---- TILE_N=8 ---------------------------------------------------
        triton.Config({'TILE_N': 8,  'BLOCK_C': 512}, num_warps=8,  num_stages=2),
        triton.Config({'TILE_N': 8,  'BLOCK_C': 512}, num_warps=16, num_stages=2),
        triton.Config({'TILE_N': 8,  'BLOCK_C': 512}, num_warps=8,  num_stages=3),
        triton.Config({'TILE_N': 8,  'BLOCK_C': 512}, num_warps=16, num_stages=3),
        # ---- TILE_N=16 (sweet spot: 3 programs/SM → 3.5 waves) ----------
        triton.Config({'TILE_N': 16, 'BLOCK_C': 512}, num_warps=8,  num_stages=2),
        triton.Config({'TILE_N': 16, 'BLOCK_C': 512}, num_warps=16, num_stages=2),
        triton.Config({'TILE_N': 16, 'BLOCK_C': 512}, num_warps=32, num_stages=2),
        triton.Config({'TILE_N': 16, 'BLOCK_C': 512}, num_warps=16, num_stages=3),
        triton.Config({'TILE_N': 16, 'BLOCK_C': 512}, num_warps=16, num_stages=4),
        triton.Config({'TILE_N': 16, 'BLOCK_C': 512}, num_warps=32, num_stages=3),
        triton.Config({'TILE_N': 16, 'BLOCK_C': 512}, num_warps=32, num_stages=4),
        # ---- TILE_N=32 --------------------------------------------------
        triton.Config({'TILE_N': 32, 'BLOCK_C': 512}, num_warps=16, num_stages=2),
        triton.Config({'TILE_N': 32, 'BLOCK_C': 512}, num_warps=32, num_stages=2),
        triton.Config({'TILE_N': 32, 'BLOCK_C': 512}, num_warps=32, num_stages=3),
        triton.Config({'TILE_N': 32, 'BLOCK_C': 512}, num_warps=32, num_stages=4),
        # ---- TILE_N=64 --------------------------------------------------
        triton.Config({'TILE_N': 64, 'BLOCK_C': 512}, num_warps=32, num_stages=2),
        triton.Config({'TILE_N': 64, 'BLOCK_C': 512}, num_warps=32, num_stages=3),
    ],
    key=['N', 'C'],
)
@triton.jit
def _fused_ln_gelu_transpose_kernel(
    input_ptr,   # [B, N, C]
    weight_ptr,  # [C]
    bias_ptr,    # [C]
    output_ptr,  # [B, C, N]
    B, N, C, eps,
    BLOCK_C: tl.constexpr,
    TILE_N:  tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    n_start = pid_n * TILE_N
    c_offs  = tl.arange(0, BLOCK_C)

    # ---- Load [TILE_N, BLOCK_C] via block-pointer (hardware-friendly) ---
    x = tl.load(
        tl.make_block_ptr(
            base=input_ptr + pid_b * N * C,
            shape=(N, C),
            strides=(C, 1),
            offsets=(n_start, 0),
            block_shape=(TILE_N, BLOCK_C),
            order=(1, 0),       # C is the contiguous (fast) dimension
        ),
        boundary_check=(0,),    # only check N boundary (C==BLOCK_C always)
        padding_option='zero',
    ).to(tl.float32)

    # ---- Layer-Norm: reduce along axis=1 (C dim) ------------------------
    mean = tl.sum(x,         axis=1) / C       # [TILE_N]
    xc   = x - mean[:, None]
    var  = tl.sum(xc * xc,   axis=1) / C       # [TILE_N]
    rstd = 1.0 / tl.sqrt(var + eps)
    xn   = xc * rstd[:, None]

    w = tl.load(weight_ptr + c_offs).to(tl.float32)
    b = tl.load(bias_ptr   + c_offs).to(tl.float32)
    y = xn * w[None, :] + b[None, :]           # [TILE_N, BLOCK_C]

    # ---- GELU (exact erf-based) -----------------------------------------
    INV_SQRT2 = 0.7071067811865476
    y_gelu = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))

    # ---- Transpose [TILE_N, BLOCK_C] → [BLOCK_C, TILE_N] ---------------
    y_t = tl.trans(y_gelu)

    # ---- Store via block-pointer (N is contiguous → coalesced writes) ---
    if IS_BF16:
        tl.store(
            tl.make_block_ptr(
                base=output_ptr + pid_b * C * N,
                shape=(C, N),
                strides=(N, 1),
                offsets=(0, n_start),
                block_shape=(BLOCK_C, TILE_N),
                order=(1, 0),   # N is the contiguous (fast) dimension
            ),
            y_t.to(tl.bfloat16),
            boundary_check=(1,),   # only check N boundary
        )
    else:
        tl.store(
            tl.make_block_ptr(
                base=output_ptr + pid_b * C * N,
                shape=(C, N),
                strides=(N, 1),
                offsets=(0, n_start),
                block_shape=(BLOCK_C, TILE_N),
                order=(1, 0),
            ),
            y_t.to(tl.float16),
            boundary_check=(1,),
        )


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_layernorm_transpose_gelu_512(in_0, in_1, in_2):
    """
    in_0 : bias   [C]
    in_1 : weight [C]
    in_2 : input  [B, N, C]
    returns       [B, C, N]
    """
    B, N, C = in_2.shape
    IS_BF16 = 1 if in_2.dtype == torch.bfloat16 else 0
    output  = torch.empty((B, C, N), dtype=in_2.dtype, device=in_2.device)

    _fused_ln_gelu_transpose_kernel[
        lambda meta: (triton.cdiv(N, meta['TILE_N']), B)
    ](
        in_2, in_1, in_0, output,
        B, N, C, 1e-5,
        IS_BF16=IS_BF16,
    )
    return output


def replacement_func():
    return fused_layernorm_transpose_gelu_512