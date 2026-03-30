import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(32, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_lss_32x64_kernel(
    in_0_ptr,   # bias  [C]
    in_1_ptr,   # weight [C, K]
    in_2_ptr,   # input  [B, K]
    in_3_ptr,   # feat   [B, C, H, W]
    out_ptr,    # output [B, C, H, W]
    C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: dim0 = b*C+c, dim1 = spatial block
    bc_pid = tl.program_id(0)
    sp_pid  = tl.program_id(1)

    b = bc_pid // C
    c = bc_pid % C

    # Compute linear: dot(in_2[b], in_1[c]) + in_0[c], then sigmoid
    k_offs = tl.arange(0, 8)
    x    = tl.load(in_2_ptr + b * 8 + k_offs).to(tl.float32)
    w    = tl.load(in_1_ptr + c * 8 + k_offs).to(tl.float32)
    bias = tl.load(in_0_ptr + c).to(tl.float32)
    scale = tl.sigmoid(tl.sum(x * w, axis=0) + bias)

    # Multiply spatial tile of in_3 by scale
    sp_offs  = sp_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask     = sp_offs < HW
    base     = bc_pid * HW
    in3_vals = tl.load(in_3_ptr + base + sp_offs, mask=mask)
    out_vals = in3_vals * scale.to(in3_vals.dtype)
    tl.store(out_ptr + base + sp_offs, out_vals, mask=mask)


@torch.fx.wrap
def fused_linear_sigmoid_mul_32_64(in_0, in_1, in_2, in_3):
    B, C, H, W = in_3.shape
    HW  = H * W
    BC  = B * C
    out = torch.empty_like(in_3)
    grid = lambda meta: (BC, triton.cdiv(HW, meta['BLOCK_SIZE']))
    _fused_lss_32x64_kernel[grid](
        in_0, in_1, in_2, in_3, out,
        C, HW,
    )
    return out


def replacement_func():
    return fused_linear_sigmoid_mul_32_64