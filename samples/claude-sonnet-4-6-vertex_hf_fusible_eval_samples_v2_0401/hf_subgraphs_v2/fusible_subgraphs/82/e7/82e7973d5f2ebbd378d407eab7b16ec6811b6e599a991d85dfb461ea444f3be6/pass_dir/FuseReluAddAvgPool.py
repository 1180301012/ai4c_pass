import torch
import triton
import triton.language as tl


# ── Pattern: matches exactly what model.py computes ──────────────────────────

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ── Triton kernel: fused relu + add + global-average-pool ────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32},  num_warps=1),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=1),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_relu_add_avgpool_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    HW,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per (batch, channel) pair.
    Accumulates relu(in_1[b,c,:,:]) + in_0[b,c,:,:] in float32,
    divides by HW, and stores one output element.
    """
    pid = tl.program_id(0)

    in0_base = in0_ptr + pid * HW
    in1_base = in1_ptr + pid * HW

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for offset in range(0, HW, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW

        # Load with other=0.0 so masked positions contribute 0 to the sum
        x0 = tl.load(in0_base + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(in1_base + offsets, mask=mask, other=0.0).to(tl.float32)

        # relu(x1) + x0; masked positions already 0 after load
        val = tl.maximum(x1, 0.0) + x0
        acc += val

    total    = tl.sum(acc, axis=0)
    mean_f32 = total / HW

    # Write back in the original dtype
    if USE_FP16:
        tl.store(out_ptr + pid, mean_f32.to(tl.float16))
    elif USE_BF16:
        tl.store(out_ptr + pid, mean_f32.to(tl.bfloat16))
    else:
        tl.store(out_ptr + pid, mean_f32)


# ── Python wrapper (must be @torch.fx.wrap) ──────────────────────────────────

@torch.fx.wrap
def fused_relu_add_avgpool(in_0, in_1):
    B, C, H, W = in_0.shape
    HW = H * W
    BC = B * C

    # Ensure contiguous NCHW layout for correct offset arithmetic
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()

    # Output tensor has the same dtype as the inputs
    out = torch.empty(BC, dtype=in_0.dtype, device=in_0.device)

    use_fp16 = (in_0.dtype == torch.float16)
    use_bf16 = (in_0.dtype == torch.bfloat16)

    _fused_relu_add_avgpool_kernel[(BC,)](
        in_0,
        in_1,
        out,
        HW,
        USE_FP16=use_fp16,
        USE_BF16=use_bf16,
    )

    # Reshape flat [B*C] → [B, C, 1, 1]
    return out.view(B, C, 1, 1)


# ── Hook for the pass framework ───────────────────────────────────────────────

def replacement_func():
    return fused_relu_add_avgpool