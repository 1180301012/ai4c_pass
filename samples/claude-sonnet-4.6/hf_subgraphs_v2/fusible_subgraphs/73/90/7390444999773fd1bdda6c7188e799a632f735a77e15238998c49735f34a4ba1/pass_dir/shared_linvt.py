import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 32,  'BLOCK_K': 32},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 64,  'BLOCK_K': 32},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 128, 'BLOCK_K': 32},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_S': 32,  'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_S': 256, 'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_S': 32,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 64,  'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_S': 256, 'BLOCK_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_S': 64,  'BLOCK_K': 128}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_S': 128, 'BLOCK_K': 128}, num_stages=5, num_warps=8),
    ],
    key=['B', 'S', 'H_in', 'nh'],
)
@triton.jit
def linear_transpose_kernel(
    hidden_ptr, weight_ptr, bias_ptr, out_ptr,
    B, S, H_in, nh,
    HD: tl.constexpr,   # head_dim, always 64
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused linear + view + transpose kernel.
    Grid: (B, ceil(S/BLOCK_S), nh)  -- no division needed for b/s indices!
    Each program handles one head for BLOCK_S sequence positions.

    hidden: [B, S, H_in]
    weight: [nh*HD, H_in]
    bias:   [nh*HD]
    out:    [B, nh, S, HD]  -- written directly in transposed layout
    """
    pid_b = tl.program_id(0)   # batch index in [0, B)
    pid_s = tl.program_id(1)   # S-block index in [0, ceil(S/BLOCK_S))
    pid_h = tl.program_id(2)   # head index in [0, nh)

    # Sequence offsets for this tile
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)   # [BLOCK_S]
    # Head-dimension offsets (always 0..HD-1)
    offs_hd = tl.arange(0, HD)                          # [HD]

    # Weight rows for this head: weight[pid_h*HD : (pid_h+1)*HD, :]
    w_row_start = pid_h * HD

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_S, HD), dtype=tl.float32)

    # Base offset into hidden for this batch (sequence offset handled via offs_s)
    h_base = pid_b * S * H_in

    for k_start in range(0, H_in, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load hidden[pid_b, offs_s, offs_k]  [BLOCK_S, BLOCK_K]
        # offs_s already includes pid_s * BLOCK_S, so h_base only needs batch offset
        h_ptrs = hidden_ptr + h_base + offs_s[:, None] * H_in + offs_k[None, :]
        h_mask = (offs_s[:, None] < S) & (offs_k[None, :] < H_in)
        h = tl.load(h_ptrs, mask=h_mask, other=0.0)

        # Load weight[w_row_start + offs_hd, offs_k]  [HD, BLOCK_K]
        w_ptrs = weight_ptr + (w_row_start + offs_hd)[:, None] * H_in + offs_k[None, :]
        w_mask = offs_k[None, :] < H_in
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # acc += h @ w^T : [BLOCK_S, BLOCK_K] @ [BLOCK_K, HD] -> [BLOCK_S, HD]
        acc = tl.dot(h, tl.trans(w), acc)

    # Add bias for this head: bias[w_row_start : w_row_start + HD]
    bias = tl.load(bias_ptr + w_row_start + offs_hd)
    acc = acc + bias[None, :]

    # Write out[pid_b, pid_h, offs_s, offs_hd] -- contiguous [BLOCK_S, HD] block
    # offs_s already includes pid_s * BLOCK_S, so out_base only needs batch+head offsets
    out_base = (pid_b * nh * S + pid_h * S) * HD
    out_ptrs = out_ptr + out_base + offs_s[:, None] * HD + offs_hd[None, :]
    out_mask = offs_s[:, None] < S
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def fused_linear_view_transpose(hidden, weight, bias, nh, hd):
    """
    Fused replacement for: linear(hidden, weight, bias) -> view(B,-1,nh,hd) -> transpose(1,2)
    Output: [B, nh, S, hd] tensor written directly, no separate transpose needed.
    hd is always 64 for all target graphs.
    """
    B, S, H_in = hidden.shape
    device = hidden.device
    dtype = hidden.dtype

    # Move weight/bias to same device and dtype as hidden if needed
    if weight.device.type == 'cpu' or weight.device != device:
        weight = weight.to(device=device, dtype=dtype)
    elif weight.dtype != dtype:
        weight = weight.to(dtype=dtype)

    if bias.device.type == 'cpu' or bias.device != device:
        bias = bias.to(device=device, dtype=dtype)
    elif bias.dtype != dtype:
        bias = bias.to(dtype=dtype)

    # Allocate output tensor in transposed layout [B, nh, S, hd]
    out = torch.empty((B, nh, S, hd), dtype=dtype, device=device)

    grid = lambda meta: (B, triton.cdiv(S, meta['BLOCK_S']), nh)

    linear_transpose_kernel[grid](
        hidden, weight, bias, out,
        B, S, H_in, nh,
        HD=hd,
    )

    return out