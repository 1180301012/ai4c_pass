"""
Shared Triton kernel for fused element-wise add + 2D spatial mean reduction.
Handles 1, 2, or 3 input tensors.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_add_mean2d_kernel(
    in0_ptr, in1_ptr, in2_ptr,
    out_ptr, mean_ptr,
    NC, HW,
    NUM_INP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW
    acc = 0.0

    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        x0 = tl.load(in0_ptr + base + offs, mask=mask, other=0.0)
        result = x0
        if NUM_INP >= 2:
            x1 = tl.load(in1_ptr + base + offs, mask=mask, other=0.0)
            result = result + x1
        if NUM_INP >= 3:
            x2 = tl.load(in2_ptr + base + offs, mask=mask, other=0.0)
            result = result + x2
        tl.store(out_ptr + base + offs, result, mask=mask)
        acc += tl.sum(result.to(tl.float32), axis=0)

    mean_val = acc / HW
    tl.store(mean_ptr + pid, mean_val)


@torch.fx.wrap
def _fused_kernel_1inp(in0):
    N, C, H, W = in0.shape
    HW = H * W
    NC = N * C
    out = torch.empty_like(in0)
    mean_buf = torch.empty((N, C, 1, 1), dtype=torch.float32, device=in0.device)
    _fused_add_mean2d_kernel[(NC,)](in0, in0, in0, out, mean_buf, NC, HW, NUM_INP=1)
    if in0.dtype == torch.float32:
        return out, mean_buf
    return out, mean_buf.to(in0.dtype)


@torch.fx.wrap
def _fused_kernel_2inp(in0, in1):
    N, C, H, W = in0.shape
    HW = H * W
    NC = N * C
    out = torch.empty_like(in0)
    mean_buf = torch.empty((N, C, 1, 1), dtype=torch.float32, device=in0.device)
    _fused_add_mean2d_kernel[(NC,)](in0, in1, in0, out, mean_buf, NC, HW, NUM_INP=2)
    if in0.dtype == torch.float32:
        return out, mean_buf
    return out, mean_buf.to(in0.dtype)


@torch.fx.wrap
def _fused_kernel_3inp(in0, in1, in2):
    N, C, H, W = in0.shape
    HW = H * W
    NC = N * C
    out = torch.empty_like(in0)
    mean_buf = torch.empty((N, C, 1, 1), dtype=torch.float32, device=in0.device)
    _fused_add_mean2d_kernel[(NC,)](in0, in1, in2, out, mean_buf, NC, HW, NUM_INP=3)
    if in0.dtype == torch.float32:
        return out, mean_buf
    return out, mean_buf.to(in0.dtype)


# Transparent (non-wrapped) shared dispatch function.
# FIXED 4-argument signature: in0, in1, in2, route
# When FX traces this, 'route' is a string constant -> the if/elif
# is resolved at trace-time, picking the correct @torch.fx.wrap call.
# The wrapped inner functions become opaque nodes; their tuple outputs
# are explicitly unpacked to produce exactly 2 graph outputs, matching
# every pattern that returns (sum_tensor, mean_tensor).
def shared_dispatch(in0, in1, in2, route):
    if route == "1":
        result = _fused_kernel_1inp(in0)
    elif route == "2":
        result = _fused_kernel_2inp(in0, in1)
    else:
        result = _fused_kernel_3inp(in0, in1, in2)
    return result[0], result[1]