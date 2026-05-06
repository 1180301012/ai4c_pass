import torch
import triton
import triton.language as tl


def _next_pow2(n):
    """Return smallest power-of-2 >= n (used to size BLOCK for triton tl.arange)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def _attn_mask_fused_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused: out[0,0,i,j] = 0 if (j<=i AND in_ptr[0,j] == 0) else -3.4e38."""
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N * N
    row = offsets // N
    col = offsets % N

    # causal mask: lower-triangle (col<=row) → attend=0, upper → -inf
    causal_ok = (col <= row).to(tl.int1)

    # input mask: in_ptr[0, col] == 0 → attend, else don't-attend
    x = tl.load(in_ptr + col, mask=mask, other=0).to(tl.int32)
    in_0_ok = (x == 0).to(tl.int1)

    # -inf when either guard fails; 0 when both attend
    both_ok = (causal_ok != 0) & (in_0_ok != 0)
    not_ok = both_ok == 0
    # tl.where bool → float(-inf), float(0): both srcs are clearly float
    out_val = tl.where(not_ok, float('-inf'), 0.0)
    tl.store(out_ptr + offsets, out_val.to(tl.float32), mask=mask)


@torch.fx.wrap
def dispatch_fused_mask(in_0):
    """
    Compute [1, 1, N, N] float32 causal+input attention mask.
    in_0 can be:
      * Original [1, N] int64  → N = in_0.shape[1]
      * Expanded  [1, 1, N, N] float32  → N = in_0.shape[2]
    We always treat in_0 as [1, N] (ignoring batch/head dims for this projection).
    """
    if in_0.dim() == 2:
        N = in_0.shape[1]
    else:
        # 4-D expanded form: shape [1, 1, N, N]
        N = int(in_0.shape[2])
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    n_elem = N * N
    BLOCK = _next_pow2(n_elem)
    _attn_mask_fused_kernel[(1,)](in_0, out, N=N, BLOCK=BLOCK)
    return out