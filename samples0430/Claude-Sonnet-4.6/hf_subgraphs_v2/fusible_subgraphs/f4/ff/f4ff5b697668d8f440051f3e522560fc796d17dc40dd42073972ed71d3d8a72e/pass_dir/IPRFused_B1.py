import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['BK'],
)
@triton.jit
def _ipr_fused_kernel(
    in2_ptr,    # [B, 17, 4096]
    in0_ptr,    # [1, 1, 1, 64]
    in1_ptr,    # [1, 1, 64, 1]
    out3_ptr,   # [B, 17, 4096] (viewed as [B, 17, 64, 64])
    out10_ptr,  # [B, 17, 2]
    BK,         # B * 17
    BLOCK_N: tl.constexpr,  # 4096
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    # Load in_2[b, k, :] (4096 elements)
    x = tl.load(in2_ptr + pid * BLOCK_N + offsets)

    # Cast to float32 for numerical stability
    x_f32 = x.to(tl.float32)

    # Numerically stable softmax
    x_max = tl.max(x_f32, axis=0)
    x_exp = tl.exp(x_f32 - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    s_f32 = x_exp / x_sum

    # Cast back to original dtype and store softmax output
    s = s_f32.to(x.dtype)
    tl.store(out3_ptr + pid * BLOCK_N + offsets, s)

    # Column index j = offset % 64, row index i = offset // 64
    j = offsets % 64
    i_idx = offsets // 64

    # Load coordinate weights (small, cached)
    x_coords = tl.load(in0_ptr + j).to(tl.float32)
    y_coords = tl.load(in1_ptr + i_idx).to(tl.float32)

    # Weighted sums
    sum_x = tl.sum(s_f32 * x_coords, axis=0)
    sum_y = tl.sum(s_f32 * y_coords, axis=0)

    # Store [b, k, 0] = sum_x, [b, k, 1] = sum_y
    tl.store(out10_ptr + pid * 2,     sum_x.to(x.dtype))
    tl.store(out10_ptr + pid * 2 + 1, sum_y.to(x.dtype))


@torch.fx.wrap
def _ipr_fused(in_0, in_1, in_2):
    B = in_2.shape[0]
    K = in_2.shape[1]   # 17
    BK = B * K

    out3  = torch.empty(B, K, 64, 64, dtype=in_2.dtype, device=in_2.device)
    out10 = torch.empty(B, K, 2,     dtype=in_2.dtype, device=in_2.device)

    _ipr_fused_kernel[(BK,)](
        in2_ptr=in_2,
        in0_ptr=in_0,
        in1_ptr=in_1,
        out3_ptr=out3,
        out10_ptr=out10,
        BK=BK,
        BLOCK_N=4096,
    )

    return out3, out10


# ── pattern ──────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    tmp_2  = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3  = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4  = tmp_3.mul(in_0)
    tmp_5  = tmp_4.reshape(1, 17, -1)
    tmp_6  = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7  = tmp_3.mul(in_1)
    tmp_8  = tmp_7.reshape(1, 17, -1)
    tmp_9  = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return tmp_3, tmp_10


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return _ipr_fused