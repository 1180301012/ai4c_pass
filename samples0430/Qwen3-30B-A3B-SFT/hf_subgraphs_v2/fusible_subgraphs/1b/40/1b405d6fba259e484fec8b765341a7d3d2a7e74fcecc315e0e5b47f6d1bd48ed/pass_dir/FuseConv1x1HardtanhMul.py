import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 1x1 conv2d + hardtanh (ReLU6) + element-wise multiply
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton fused kernel
#
# Computes for every (batch b, spatial position hw, output channel co):
#   acc = sum_ci( weight[co, ci] * input[b, ci, hw] ) + bias[co]
#   out[b, co, hw] = clamp_to_0_6(acc) * mask[b, co, hw]
#
# Grid:  (ceil(N / BLOCK_M),  ceil(C_out / BLOCK_N))
#  where N = B * H * W
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # Small BLOCK_HW for batch=1 cases (more programs → better occupancy)
        triton.Config({'BLOCK_HW': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_HW': 1024,'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_HW': 1024,'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # BLOCK_K=16 variants
        triton.Config({'BLOCK_HW': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['B', 'C_out', 'HW'],
)
@triton.jit
def _fused_conv1x1_ht_mul_kernel(
    input_ptr,    # [B, C_in, H, W]  NCHW
    weight_ptr,   # [C_out, C_in, 1, 1]
    bias_ptr,     # [C_out]
    mask_ptr,     # [B, C_out, H, W]  NCHW
    out_ptr,      # [B, C_out, H, W]  NCHW
    B, C_in, C_out, HW,
    BLOCK_HW: tl.constexpr,
    BLOCK_N:  tl.constexpr,
    BLOCK_K:  tl.constexpr,
):
    """
    Grid: (B * ceil(HW/BLOCK_HW),  ceil(C_out/BLOCK_N))
    Each program computes [BLOCK_N output channels, BLOCK_HW spatial positions].

    Input layout: [B, C_in, H, W] NCHW.
    For each spatial position hw in [hw_start, hw_start+BLOCK_HW) and batch b:
      x[ci, hw_local] at  input_ptr + b*C_in*HW + ci*HW + hw_local   (stride HW in ci dim)
    """
    pid_hw = tl.program_id(0)   # (b, hw_block) combined
    pid_n  = tl.program_id(1)   # c_out block

    # Decompose pid_hw → (batch idx, hw block start)
    n_hw_blocks = tl.cdiv(HW, BLOCK_HW)
    b_idx      = pid_hw // n_hw_blocks
    hw_blk     = pid_hw %  n_hw_blocks
    hw_start   = hw_blk * BLOCK_HW

    c_out_start = pid_n * BLOCK_N

    hw_offs    = hw_start   + tl.arange(0, BLOCK_HW)   # [BLOCK_HW]
    c_out_offs = c_out_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    hw_mask = hw_offs   < HW
    co_mask = c_out_offs < C_out

    hw_local = hw_offs % HW    # handle case where HW is not divisible by BLOCK_HW

    # Accumulator [BLOCK_N, BLOCK_HW] in float32
    acc = tl.zeros((BLOCK_N, BLOCK_HW), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        c_in_offs = k_start + tl.arange(0, BLOCK_K)
        cin_mask  = c_in_offs < C_in

        # ---- Load input [BLOCK_K, BLOCK_HW] ----
        # in_offs[ci, hw] = b*C_in*HW + ci*HW + hw_local
        # Last dim (hw) has stride 1 → COALESCED loads within each ci row
        in_offs = b_idx * (C_in * HW) + c_in_offs[:, None] * HW + hw_offs[None, :]
        x = tl.load(input_ptr + in_offs,
                    mask=cin_mask[:, None] & hw_mask[None, :],
                    other=0.0).to(tl.float32)   # [BLOCK_K, BLOCK_HW]

        # ---- Load weight [BLOCK_N, BLOCK_K] ----
        # weight[co, ci]  at  co*C_in + ci
        # Last dim (ci) has stride 1 → COALESCED
        w_offs = c_out_offs[:, None] * C_in + c_in_offs[None, :]
        w = tl.load(weight_ptr + w_offs,
                    mask=co_mask[:, None] & cin_mask[None, :],
                    other=0.0).to(tl.float32)   # [BLOCK_N, BLOCK_K]

        # acc += w @ x  :  [BLOCK_N, BLOCK_K] x [BLOCK_K, BLOCK_HW] → [BLOCK_N, BLOCK_HW]
        acc += tl.dot(w, x)

    # Bias broadcast over spatial dim
    bias = tl.load(bias_ptr + c_out_offs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias[:, None]

    # Hardtanh clamp (ReLU6): max(0, min(6, acc))
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    # Load mask [BLOCK_N, BLOCK_HW]  (mask[b, co, hw])
    mask_offs = b_idx * (C_out * HW) + c_out_offs[:, None] * HW + hw_offs[None, :]
    mask_val = tl.load(mask_ptr + mask_offs,
                       mask=co_mask[:, None] & hw_mask[None, :],
                       other=0.0).to(tl.float32)
    acc = acc * mask_val

    # Store output
    out_offs = b_idx * (C_out * HW) + c_out_offs[:, None] * HW + hw_offs[None, :]
    tl.store(out_ptr + out_offs, acc,
             mask=co_mask[:, None] & hw_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel wrapper (must be @torch.fx.wrap so FX doesn't trace into it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_conv1x1_ht_mul(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [C_out]
    in_1 : weight [C_out, C_in, 1, 1]
    in_2 : input  [B, C_in, H, W]
    in_3 : mask   [B, C_out, H, W]
    """
    device = in_2.device
    dtype  = in_2.dtype

    B, C_in, H, W = in_2.shape
    C_out = in_1.shape[0]
    HW    = H * W

    out = torch.empty_like(in_3)

    # Grid: (B * ceil(HW/BLOCK_HW),  ceil(C_out/BLOCK_N))
    # Note: weight in_1 and bias in_0 are always on CUDA for GPU inference.
    # Avoid torch.as_tensor to reduce Python overhead on the critical path.
    grid = lambda meta: (
        B * triton.cdiv(HW, meta['BLOCK_HW']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    _fused_conv1x1_ht_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, C_in, C_out, HW,
    )

    return out


def replacement_func():
    return fused_conv1x1_ht_mul