"""Shared kernels and dispatch wrapper for all passes in this problem."""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: fused broadcast mul + broadcast add
# Grid: (B, T) – 2D launch, no integer division
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_mul_add(
    in0_ptr,   # [2, C]
    in1_ptr,   # [1, 1, 2, C]
    in2_ptr,   # [B, T, 1, C]
    out_ptr,   # [B, T, 2, C]
    T,
    C: tl.constexpr,
):
    b_idx = tl.program_id(0)
    t_idx = tl.program_id(1)
    k = tl.arange(0, C)

    in2_val = tl.load(in2_ptr + b_idx * T * C + t_idx * C + k)
    in1_0   = tl.load(in1_ptr + k)
    in1_1   = tl.load(in1_ptr + C + k)
    in0_0   = tl.load(in0_ptr + k)
    in0_1   = tl.load(in0_ptr + C + k)

    tmp0 = in2_val * in1_0 + in0_0
    tmp1 = in2_val * in1_1 + in0_1

    out_base = out_ptr + b_idx * T * 2 * C + t_idx * 2 * C
    tl.store(out_base + k,       tmp0)
    tl.store(out_base + C + k,   tmp1)


# ---------------------------------------------------------------------------
# Kernel B: transpose [B, T, C] → [B, C, T]  (unbind i=1 slice)
# Grid: (B, T) – one program per (b, t) pair
# x[b, t, k] = x_ptr[b*T*2C + t*2C + C + k]   (i=1 slice offset)
# out[b, k, t] = out_ptr[b*C*T + k*T + t]
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_transpose(
    x_ptr,     # [B, T, C] strided (i=1 slice of [B, T, 2, C])
    out_ptr,   # [B, C, T] contiguous
    T,
    C: tl.constexpr,
):
    b_idx = tl.program_id(0)
    t_idx = tl.program_id(1)
    k = tl.arange(0, C)

    x_vals = tl.load(x_ptr + b_idx * T * 2 * C + t_idx * 2 * C + C + k)
    tl.store(out_ptr + b_idx * C * T + k * T + t_idx, x_vals)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper  (SAME function returned by ALL pass files)
# Route string is always the LAST element of args.
#   "mul_add"  → a=in_0, b=in_1, c=in_2
#   "transpose"→ a=x   (single tensor)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _dispatch(*args):
    route = args[-1]
    if route == "mul_add":
        in_0, in_1, in_2 = args[0], args[1], args[2]
        B = in_2.shape[0]
        T = in_2.shape[1]
        C = 128
        out = torch.empty((B, T, 2, C), dtype=in_2.dtype, device=in_2.device)
        _kernel_mul_add[(B, T)](in_0, in_1, in_2, out, T, C, num_warps=4)
        return out
    else:   # route == "transpose"
        # args[0] is the tuple returned by torch.unbind(a0, 2).
        # args[0][1] = i=1 slice, shape [B, T, C] — a proper tensor.
        s = args[0][1]
        B = s.shape[0]
        T = s.shape[1]
        C = s.shape[2]
        out = torch.empty((B, C, T), dtype=s.dtype, device=s.device)
        _kernel_transpose[(B, T)](s, out, T, C, num_warps=4)
        return out