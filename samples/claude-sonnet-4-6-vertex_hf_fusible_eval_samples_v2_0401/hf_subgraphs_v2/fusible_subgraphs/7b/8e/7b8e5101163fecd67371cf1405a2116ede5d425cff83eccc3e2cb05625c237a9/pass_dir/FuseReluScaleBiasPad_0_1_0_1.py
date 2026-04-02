import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: relu(in_2) * in_1 + in_0  then pad(…, (0,1,0,1), 'constant', None)
# ─────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['N_out'],
)
@triton.jit
def _fused_relu_scale_bias_pad_kernel(
    in0_ptr,          # bias  – shape [1]
    in1_ptr,          # scale – shape [1]
    in2_ptr,          # input – shape [B, C, H, W]
    out_ptr,          # output– shape [B, C, H+1, W+1]
    C, H, W,
    H_out, W_out,
    CHW, HW,          # C*H*W and H*W (input strides by batch/channel)
    CH_outW_out,      # C*H_out*W_out  (output stride by batch)
    H_outW_out,       # H_out*W_out    (output stride by channel)
    N_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    offs = base + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_out

    # ── decompose flat output index ───────────────────────────────────────
    w_out = offs % W_out
    tmp   = offs // W_out
    h_out = tmp  % H_out
    tmp2  = tmp  // H_out
    c_idx = tmp2 % C
    b_idx = tmp2 // C

    # ── load scalars (shape-[1] tensors → single element) ─────────────────
    scale = tl.load(in1_ptr).to(tl.float32)
    bias  = tl.load(in0_ptr).to(tl.float32)

    # ── which positions are inside the original (non-padded) area? ────────
    in_orig = (h_out < H) & (w_out < W)

    # ── safe input index (clamped so we never actually OOB-load) ──────────
    h_safe = tl.minimum(h_out, H - 1)
    w_safe = tl.minimum(w_out, W - 1)
    in_idx = b_idx * CHW + c_idx * HW + h_safe * W + w_safe

    x = tl.load(in2_ptr + in_idx, mask=mask & in_orig, other=0.0).to(tl.float32)

    # ── fused compute ──────────────────────────────────────────────────────
    relu_x = tl.maximum(x, 0.0)
    result = relu_x * scale + bias
    result = tl.where(in_orig, result, 0.0)

    # ── store (cast back to original dtype via pointer) ────────────────────
    tl.store(out_ptr + offs, result, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Python wrapper (must be @torch.fx.wrap so FX doesn't trace into it)
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_relu_scale_bias_pad(in_0, in_1, in_2):
    B, C, H, W = in_2.shape
    H_out = H + 1
    W_out = W + 1
    N_out = B * C * H_out * W_out

    out = torch.empty((B, C, H_out, W_out), dtype=in_2.dtype, device=in_2.device)

    grid = lambda meta: ((N_out + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_relu_scale_bias_pad_kernel[grid](
        in_0, in_1, in_2, out,
        C, H, W,
        H_out, W_out,
        C * H * W,          # CHW
        H * W,              # HW
        C * H_out * W_out,  # CH_outW_out
        H_out * W_out,      # H_outW_out
        N_out,
    )

    return out


def replacement_func():
    return fused_relu_scale_bias_pad