import torch
import triton
import triton.language as tl


@triton.jit
def compute_combined_mask_kernel(
    in0_ptr,    # [B, N] int64 - attention mask (already on cuda)
    in2_ptr,    # [S] int64 - cache positions
    out_ptr,    # [B * S * N] bool - output (logically [B, S, N])
    B, S, N,
    BLOCK_N: tl.constexpr,
):
    """
    Computes:
      out[b, s, n] = bool(in0[b, n]) & (n <= in2[s])

    which corresponds to:
      tmp13 = (arange(N) <= cache_pos[:, None]) * in0_bool
    broadcasted to shape [B, 1, S, N]
    """
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    n_offs = tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Load cache position for this row
    cache_pos = tl.load(in2_ptr + s)

    # Load in0[b, :] int64 and convert to bool
    in0_vals = tl.load(in0_ptr + b * N + n_offs, mask=n_mask, other=0)
    in0_bool = in0_vals != 0

    # Causal mask: column index <= cache_pos
    causal = n_offs <= cache_pos

    # Combined mask (bool AND)
    result = causal & in0_bool

    # Store to [b, s, n] with stride [S*N, N, 1]
    out_offs = (b * S + s) * N + n_offs
    tl.store(out_ptr + out_offs, result, mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
        triton.Config({'BLOCK': 1024}),
    ],
    key=['total'],
)
@triton.jit
def cast_to_float32_kernel(
    in_ptr,
    out_ptr,
    total,
    BLOCK: tl.constexpr,
):
    """Element-wise cast to float32."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x.to(tl.float32), mask=mask)


@torch.fx.wrap
def fused_mask_freq_prep(in_0, in_1, in_2, in_3, route):
    """
    Fused implementation of:
      tmp13 = combined causal + attention mask, shape [B, 1, S, N]
      tmp21 = in_1.float().view(1, F, 1)
      tmp22 = in_3.float().view(B3, 1, N)
    """
    B = in_0.shape[0]
    N = in_0.shape[1]
    S = in_2.shape[0]
    F = in_1.shape[0]
    B3 = in_3.shape[0]

    # ---- Kernel 1: Compute combined mask [B, S, N] ----
    out_13_flat = torch.empty((B * S * N,), dtype=torch.bool, device=in_0.device)
    BLOCK_N = triton.next_power_of_2(N)
    compute_combined_mask_kernel[(B * S,)](
        in_0, in_2, out_13_flat,
        B, S, N,
        BLOCK_N=BLOCK_N,
    )
    out_13 = out_13_flat.view(B, 1, S, N)

    # ---- Kernel 2: Cast in_1 (inv_freq) to float32 ----
    out_21_flat = torch.empty((F,), dtype=torch.float32, device=in_1.device)
    cast_to_float32_kernel[(triton.cdiv(F, 128),)](
        in_1, out_21_flat, F,
    )
    out_21 = out_21_flat.view(1, F, 1)

    # ---- Kernel 3: Cast in_3 (position_ids) to float32 ----
    total_n3 = B3 * N
    out_22_flat = torch.empty((total_n3,), dtype=torch.float32, device=in_3.device)
    cast_to_float32_kernel[(triton.cdiv(total_n3, 128),)](
        in_3, out_22_flat, total_n3,
    )
    out_22 = out_22_flat.view(B3, 1, N)

    return (out_13, out_21, out_22)