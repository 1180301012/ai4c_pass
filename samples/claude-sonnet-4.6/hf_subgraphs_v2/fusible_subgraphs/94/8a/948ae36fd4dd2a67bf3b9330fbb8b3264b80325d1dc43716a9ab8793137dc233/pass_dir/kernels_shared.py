"""
Shared Triton kernels: separate arange-fill and int64→bool cast kernels.
No autotune – fixed BLOCK sizes to avoid compilation overhead on tiny tensors.
All input shapes have numel() that is a multiple of 1024.
"""
import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Kernel 1: fill output with 0, 1, 2, ..., BLOCK-1  (arange)
# Single program, no mask, BLOCK == N exactly (N is power of 2).
# -----------------------------------------------------------------------
@triton.jit
def _arange_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, offs.to(tl.int64))


def run_arange(dev, N):
    ar_out = torch.empty(N, dtype=torch.int64, device=dev)
    _arange_kernel[(1,)](ar_out, BLOCK=N)          # N is power-of-2 constexpr
    return ar_out


# -----------------------------------------------------------------------
# Kernel 2: int64 → int8 element-wise cast (no mask – M % BLOCK == 0)
# We write int8 and then view-as-bool (zero copy) to avoid tl.int1 issues.
# -----------------------------------------------------------------------
@triton.jit
def _cast_i8_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    tl.store(out_ptr + offs, (x != 0).to(tl.int8))


def run_cast_bool(in_0):
    M = in_0.numel()
    # Allocate int8 with same shape as in_0 (avoids a reshape view)
    cast_i8 = torch.empty(in_0.shape, dtype=torch.int8, device=in_0.device)
    _cast_i8_kernel[(M // 1024,)](in_0, cast_i8, BLOCK=1024)
    # zero-copy dtype reinterpret: int8 → bool (same shape, same 1-byte storage)
    return cast_i8.view(torch.bool)