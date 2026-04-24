import torch
import triton
import triton.language as tl


# ── Pattern: exact mirror of model.py ──────────────────────────────────────
def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7


def replacement_args(in_0):
    return (in_0,)


# ── Fused Triton kernel: float32 accumulation, native-dtype output ─────────
# Hardcoded best config for A30 (NVIDIA A30, 56 SMs, 24 MB L2)
# BLOCK_SIZE=8192, num_warps=16 → 512 threads/block, 16 elems/thread
# Register analysis: ~36 regs/thread × 512 = 18432 regs/block
#   → 3 blocks/SM (thread-limited: 2048/512=4, register-limited: 65536/18432≈3)
#   → 3 × 16 warps = 48 warps/SM = 100% warp occupancy
@triton.jit
def _gelu_fused_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load in native dtype (float16 or bfloat16), upcast for computation
    x      = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32  = x.to(tl.float32)

    # GELU: 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x³)))
    x3     = x_f32 * x_f32 * x_f32
    inner  = x_f32 + 0.044715 * x3
    inner_scaled = 0.7978845608028654 * inner
    # Hardware-accelerated tanh via CUDA libdevice
    tanh_val = tl.extra.cuda.libdevice.tanh(inner_scaled)
    gelu   = 0.5 * x_f32 * (1.0 + tanh_val)

    # Store result back in the original dtype
    tl.store(out_ptr + offsets, gelu.to(x.dtype), mask=mask)


# ── Kernel wrapper (must be @torch.fx.wrap) ────────────────────────────────
@torch.fx.wrap
def gelu_fused(x):
    N   = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 8192
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    _gelu_fused_kernel[(num_programs,)](
        x, out, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )
    return out


# ── Replacement entry-point ───────────────────────────────────────────────
def replacement_func():
    return gelu_fused