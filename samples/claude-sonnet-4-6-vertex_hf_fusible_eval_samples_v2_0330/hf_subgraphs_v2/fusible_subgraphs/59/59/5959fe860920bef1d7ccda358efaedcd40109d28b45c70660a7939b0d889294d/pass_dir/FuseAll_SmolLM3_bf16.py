"""
Combined pass: fuses the full SmolLM3 subgraph:
  - RoPE:   cat(in_1, in_1, dim=-1) → cos/sin → ×1.0 → to(bfloat16)
  - RMSNorm: in_2 → to(fp32) → pow(2) → mean → +1e-6 → rsqrt → * → to(bf16) → *weight
Returns (cos_out, normed_out, sin_out) matching the model's return order.
Optimizations:
  - RMSNorm: D=2048 hardcoded as BLOCK_D constexpr → no masking, no D argument
  - RoPE: BLOCK_H = next_power_of_2(H) → no masking when H is power of 2
  - Autotune over num_warps for best performance on A30
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# RoPE kernel  (bfloat16 output) — mask-free when H is power of 2
# ---------------------------------------------------------------------------
@triton.jit
def _rope_bf16_kernel(
    in_ptr,
    cos_ptr,
    sin_ptr,
    H,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    x = tl.load(in_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    c = tl.cos(x).to(tl.bfloat16)
    s = tl.sin(x).to(tl.bfloat16)
    out_base = row * 2 * H
    tl.store(cos_ptr + out_base + cols,     c, mask=mask)
    tl.store(cos_ptr + out_base + H + cols, c, mask=mask)
    tl.store(sin_ptr + out_base + cols,     s, mask=mask)
    tl.store(sin_ptr + out_base + H + cols, s, mask=mask)


# ---------------------------------------------------------------------------
# RMSNorm kernel  (eps=1e-6, bfloat16 output)
# Autotune key is D (always 2048) so config is cached after first graph.
# Reduced to 2 configs (num_warps 8 and 16) — best for A30 with D=2048.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 2048}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_D': 2048}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_D': 2048}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_D': 2048}, num_warps=32, num_stages=1),
    ],
    key=['D'],
)
@triton.jit
def _rms_norm_1e6_bf16_v2_kernel(
    x_ptr, w_ptr, out_ptr,
    D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x   = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    x2  = x * x
    mean_x2 = tl.sum(x2, axis=0) * (1.0 / D)
    rstd = tl.rsqrt(mean_x2 + 1e-6)
    x_hat = x * rstd
    w   = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    out = (x_hat * w).to(tl.bfloat16)
    tl.store(out_ptr + row * D + cols, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapped sub-kernels  (opaque to FX — each returns ONE tensor or a tuple)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _smollm3_rope_bf16(in_1):
    """Runs RoPE, returns (cos_out, sin_out) tuple in bfloat16.
    H is always 64 for all SmolLM3 graphs → BLOCK_H hardcoded to 64."""
    rope_shape = in_1.shape
    H = rope_shape[-1]
    N_rope = in_1.numel() // H
    out_rope_shape = list(rope_shape[:-1]) + [2 * H]
    cos_out = torch.empty(out_rope_shape, dtype=torch.bfloat16, device=in_1.device)
    sin_out = torch.empty(out_rope_shape, dtype=torch.bfloat16, device=in_1.device)
    # H is always 64 for SmolLM3; hardcoding avoids triton.next_power_of_2 overhead
    _rope_bf16_kernel[(N_rope,)](in_1, cos_out, sin_out, H, BLOCK_H=64)
    return cos_out, sin_out


@torch.fx.wrap
def _smollm3_rmsnorm_bf16(in_0, in_2):
    """Runs RMSNorm (eps=1e-6, bf16 output), returns normed tensor."""
    norm_shape = in_2.shape
    D = norm_shape[-1]
    N_norm = in_2.numel() // D
    x_2d = in_2.view(N_norm, D)
    out_2d = torch.empty_like(x_2d)
    _rms_norm_1e6_bf16_v2_kernel[(N_norm,)](x_2d, in_0, out_2d, D)
    return out_2d.view(norm_shape)


# ---------------------------------------------------------------------------
# Outer replacement function  — NOT @torch.fx.wrap so FX can trace into it.
# FX sees: getitem(rope_result,0), getitem(rope_result,1), rmsnorm_result
# → 3 separate returning nodes, matching the pattern's 3 return nodes.
# ---------------------------------------------------------------------------
def _smollm3_combined_bf16(in_0, in_1, in_2):
    rope_result = _smollm3_rope_bf16(in_1)
    cos_out = rope_result[0]
    sin_out = rope_result[1]
    normed  = _smollm3_rmsnorm_bf16(in_0, in_2)
    return cos_out, normed, sin_out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    # RoPE branch
    tmp_1  = torch.cat((in_1, in_1), dim=-1)
    tmp_2  = tmp_1.cos()
    tmp_3  = tmp_2 * 1.0
    tmp_4  = tmp_1.sin()
    tmp_5  = tmp_4 * 1.0
    tmp_6  = tmp_3.to(dtype=torch.bfloat16)
    tmp_7  = tmp_5.to(dtype=torch.bfloat16)
    # RMSNorm branch
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_6, tmp_17, tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return _smollm3_combined_bf16