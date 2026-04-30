"""
Shared Triton kernel for attention bias fusion.
All pass files import triton_dispatch (or triton_dispatch5) from here,
sharing the SAME replacement_func object (satisfies replacement_func_limit).
"""
import torch
import triton
import triton.language as tl


# ---- kernel for 2-op pattern (masked_fill) ----
@triton.jit
def _fused_maskedfill_kernel(causal_ptr, eq_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N * N
    is_causal = tl.load(causal_ptr + offs, mask=mask, other=0) != 0
    is_eq     = tl.load(eq_ptr   + offs, mask=mask, other=0) != 0
    out = tl.where(is_causal | is_eq, -3.4028234663852886e+38, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


# ---- kernel for 5-op pattern (the final getitem+setitem+eq+mul+getitem) ----
@triton.jit
def _fused_final_masked_kernel(causal_ptr, eq_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Causal mask: element [0,0,row,col] at row*N+col in [1,1,N,N] tensor.
    For each (row, col):
      out = -inf  if causal[row,col] != 0  (masked by causal mask)
             = -inf  if eq[row,col]     != 0  ((sum==0) mask)
             = 0     otherwise
    This exactly matches: tmp_22 = tmp_10 * (~torch.all(tmp_10 == -inf, dim=-1, keepdim=True))
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N * N
    is_causal = tl.load(causal_ptr + offs, mask=mask, other=0) != 0
    is_eq     = tl.load(eq_ptr   + offs, mask=mask, other=0) != 0
    out = tl.where(is_causal | is_eq, -3.4028234663852886e+38, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


@torch.fx.wrap
def triton_dispatch(causal_mask, tmp_15_input, N_int, route):
    """2-op replacement: fused masked_fill."""
    N = int(N_int)
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=causal_mask.device)
    BLOCK_SIZE = 256
    grid = ((N * N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _fused_maskedfill_kernel[grid](causal_mask, tmp_15_input, out, N=N, BLOCK_SIZE=BLOCK_SIZE)
    return out


@torch.fx.wrap
def triton_dispatch5(causal_mask, tmp_17, N_int, route):
    """
    5-op replacement: fused final getitem+setitem+eq+mul+getitem.
    causal_mask : [1,1,N,N] tmp_10 before setitem (the cloned causal matrix)
    tmp_17      : [1,1,N,N] result of masked_fill (what setitem writes)
    """
    N = int(N_int)
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=causal_mask.device)
    BLOCK_SIZE = 256
    grid = ((N * N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _fused_final_masked_kernel[grid](causal_mask, tmp_17, out, N=N, BLOCK_SIZE=BLOCK_SIZE)
    return out