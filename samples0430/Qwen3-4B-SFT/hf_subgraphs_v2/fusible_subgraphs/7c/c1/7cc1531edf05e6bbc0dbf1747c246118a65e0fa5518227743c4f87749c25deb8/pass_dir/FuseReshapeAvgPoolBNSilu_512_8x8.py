import torch
import triton
import triton.language as tl


# ── Pattern ─────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Matches:
        reshape(1, 512, 16, 16) → avg_pool2d → batch_norm → silu
    All arguments must match model.py exactly (positional ops, same order).
    """
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ── Triton kernel ────────────────────────────────────────────────────────────
# Fuses: reshape + avg_pool2d(kernel=2,stride=2) + batch_norm(inference) + silu
#
# Grid layout:
#   program_id(0) ∈ [0, C_total)  → channel index (C_total = B*C_out = 512)
#   program_id(1) ∈ [0, HW // BLOCK_SIZE)  → block of output positions
#
# One block handles BLOCK_SIZE consecutive output positions (w_idx * OH * OW,
# w_idx * OH * OW + ...) for a fixed (B, C_out) pair.
#
# Average-pool input indexing for [1,512,16,16]:
#   in_ptr[ c_id * CH_input  +  row * W_in_fc  +  col ]
#   where (row,h)=(h*8+w), col ∈ {2*w, 2*w+1}, W_in_fc=16

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
    ],
    key=['C_total'],
)
@triton.jit
def _fused_avgpool_bn_silu_kernel(
    in4_ptr,           # [1, C_total, H_in,  W_in]  contiguous
    mean_ptr,          # [C_total]  running_mean
    var_ptr,           # [C_total]  running_var
    weight_ptr,        # [C_total]  scale
    bias_ptr,          # [C_total]  bias
    out_ptr,           # [1, C_total, OH, OW]
    C_total,           # = B * C_out  (= 512)
    CH_input,          # = H_in * W_in  (= 256)
    H_in_fc,           # = H_in  (= 16)
    W_in_fc,           # = W_in  (= 16)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    c_id  = tl.program_id(0)
    bid   = tl.program_id(1)

    pos_start  = bid * BLOCK_SIZE
    positions  = pos_start + tl.arange(0, BLOCK_SIZE)   # flat OH*OW index
    mask       = positions < (H_in_fc * W_in_fc)  # 64

    # Decompose flat position  →  (h_ave, w_ave)
    w_idx = positions % W_in_fc   # width output index  [0, 8)
    h_idx = positions // W_in_fc  # height output index [0, 8)

    # ── Load per-channel BN parameters ─────────────────────────────────────
    mean_val = tl.load(mean_ptr   + c_id).to(tl.float32)
    var_val  = tl.load(var_ptr    + c_id).to(tl.float32)
    w_val    = tl.load(weight_ptr + c_id).to(tl.float32)
    b_val    = tl.load(bias_ptr   + c_id).to(tl.float32)

    CH_stride = C_total * CH_input
    H_stride  = H_in_fc * W_in_fc
    W_stride  = W_in_fc        # = 16

    # ── Average-pool (4-element window, stride=2) ──────────────────────────
    # base offset for in4[c_id, h_idx, w_idx]
    base = c_id * CH_input + h_idx * W_stride + w_idx
    v00 = tl.load(in4_ptr + base,     mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(in4_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(in4_ptr + base + H_stride,          mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(in4_ptr + base + H_stride + 1,      mask=mask, other=0.0).to(tl.float32)
    avg = (v00 + v01 + v10 + v11) * (1.0 / 4.0)

    # ── Batch-norm (inference: training=False) ──────────────────────────────
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    x_norm  = (avg - mean_val) * inv_std
    bn_out  = x_norm * w_val + b_val

    # ── SiLU ────────────────────────────────────────────────────────────────
    silu_out = bn_out * tl.sigmoid(bn_out)

    # ── Store output ────────────────────────────────────────────────────────
    out_pos = c_id * H_in_fc * W_in_fc + h_idx * W_in_fc + w_idx
    tl.store(out_ptr + out_pos, silu_out.to(out_ptr.dtype.element_ty), mask=mask)


# ── Kernel wrapper ───────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_avgpool_bn_silu(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : running_mean  [C]       torch.bfloat16 / float16
    in_1 : running_var   [C]
    in_2 : bias          [C]
    in_3 : weight        [C]
    in_4 : input         [B, C, H, W]  on CUDA
    """
    B  = in_4.shape[0]
    C_out   = in_4.shape[1]
    H_in    = in_4.shape[2]
    W_in    = in_4.shape[3]
    C_total = B * C_out                                    # 512
    CH_input  = H_in * W_in                               # 256

    out = torch.empty((B, C_out, H_in // 2, W_in // 2),
                      dtype=in_4.dtype, device=in_4.device)

    grid = lambda meta: (C_total, triton.cdiv(CH_input, meta['BLOCK_SIZE']))

    _fused_avgpool_bn_silu_kernel[grid](
        in_4,          # in4_ptr
        in_0,          # mean_ptr
        in_1,          # var_ptr
        in_3,          # weight_ptr
        in_2,          # bias_ptr
        out,           # out_ptr
        C_total,
        CH_input,
        H_in,          # H_in_fc
        W_in,          # W_in_fc
        1e-5,          # eps (constexpr)
    )

    return out


# ── Replacement function ─────────────────────────────────────────────────────
def replacement_func():
    return fused_avgpool_bn_silu