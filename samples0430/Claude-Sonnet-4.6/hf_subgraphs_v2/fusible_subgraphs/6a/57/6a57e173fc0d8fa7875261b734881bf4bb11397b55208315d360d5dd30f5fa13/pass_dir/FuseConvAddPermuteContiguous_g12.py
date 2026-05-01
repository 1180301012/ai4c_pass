"""
Fused pass: torch.conv2d(groups=12, kernel=65x1, pad=32) + in_place_add + permute(0,2,1,3) + contiguous

Matches the ogoshi2000/stance-nystromformer variant with 12 attention heads.
"""

import torch
import triton
import triton.language as tl
import operator
import torch.fx

# Patch Proxy.__iadd__ to emit operator.iadd (matching torchdynamo)
def _proxy_iadd(self, other):
    return self.tracer.create_proxy('call_function', operator.iadd, (self, other), {})

torch.fx.Proxy.__iadd__ = _proxy_iadd


# ---------------------------------------------------------------------------
# Pattern  (groups=12)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    conv_out = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 12)
    in_1 += conv_out
    permuted = in_1.permute(0, 2, 1, 3)
    cont     = permuted.contiguous()
    return cont


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel  (identical logic to g4, but separate compiled instance)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Groups=12 always has D=64 → only BLOCK_D=64 configs needed
        triton.Config({'BLOCK_S':  2, 'BLOCK_D': 64}, num_warps=2),
        triton.Config({'BLOCK_S':  4, 'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_S':  8, 'BLOCK_D': 64}, num_warps=8),
        triton.Config({'BLOCK_S': 16, 'BLOCK_D': 64}, num_warps=8),
    ],
    key=['S'],
)
@triton.jit
def _fused_conv_add_permute_g12_kernel(
    in2_ptr,
    w_ptr,
    in1_ptr,
    out_ptr,
    B, G, S, D,
    GSD, SD, GD,
    K,       # = 65 (runtime)
    PADDING, # = 32 (runtime)
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bg = tl.program_id(0)
    pid_s  = tl.program_id(1)
    pid_d  = tl.program_id(2)

    g = pid_bg % G
    b = pid_bg // G

    s_start = pid_s * BLOCK_S
    d_start = pid_d * BLOCK_D

    s_ids = s_start + tl.arange(0, BLOCK_S)
    d_ids = d_start + tl.arange(0, BLOCK_D)

    s_mask = s_ids < S
    d_mask = d_ids < D
    mask   = s_mask[:, None] & d_mask[None, :]

    acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    for k in range(K):
        w_val = tl.load(w_ptr + g * K + k)
        k_off = k - PADDING
        in_s  = s_ids + k_off
        valid = (in_s >= 0) & (in_s < S) & s_mask

        in2_off = b * GSD + g * SD + in_s[:, None] * D + d_ids[None, :]
        vals = tl.load(in2_ptr + in2_off,
                       mask=valid[:, None] & d_mask[None, :], other=0.0)
        acc = acc + w_val * vals.to(tl.float32)

    in1_off  = b * GSD + g * SD + s_ids[:, None] * D + d_ids[None, :]
    in1_vals = tl.load(in1_ptr + in1_off, mask=mask, other=0.0)
    acc = acc + in1_vals.to(tl.float32)

    out_off = b * GSD + s_ids[:, None] * GD + g * D + d_ids[None, :]
    tl.store(out_ptr + out_off, acc.to(in1_vals.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv_add_permute_g12(in_0, in_1, in_2):
    B, G, S, D = in_1.shape
    out = torch.empty((B, S, G, D), dtype=in_1.dtype, device=in_1.device)

    def grid(meta):
        return (
            B * G,
            triton.cdiv(S, meta['BLOCK_S']),
            triton.cdiv(D, meta['BLOCK_D']),
        )

    _fused_conv_add_permute_g12_kernel[grid](
        in_2, in_0, in_1, out,
        B, G, S, D,
        G * S * D,
        S * D,
        G * D,
        65,   # K (runtime int)
        32,   # PADDING (runtime int)
    )
    return out


def replacement_func():
    return fused_conv_add_permute_g12