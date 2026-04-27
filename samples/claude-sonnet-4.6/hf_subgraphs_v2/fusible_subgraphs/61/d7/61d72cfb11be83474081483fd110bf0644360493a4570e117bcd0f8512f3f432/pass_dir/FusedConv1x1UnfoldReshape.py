import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['C_out', 'C_in', 'N_out'],
)
@triton.jit
def fused_conv1x1_unfold_kernel(
    input_ptr,   # [1, C_in, H, W] — feature map
    weight_ptr,  # [C_out, C_in, 1, 1] — conv weight
    output_ptr,  # [1, C_out, 4, P_BLOCKS] — final fused output
    C_out,
    C_in,
    H,
    W,
    P_BLOCKS,    # = (H//2) * (W//2)
    N_BLK_W,     # = W // 2
    N_out,       # = 4 * P_BLOCKS
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel:
      output[0, c, k, p] = sum_{cin} weight[c, cin] * input[0, cin, h, w]
    where:
      k  in [0, 4)           — position within 2x2 unfold kernel
      p  in [0, P_BLOCKS)    — block index (row-major over 2x2 blocks)
      h  = (p // N_BLK_W)*2 + k//2
      w  = (p %  N_BLK_W)*2 + k% 2

    Treated as gemm: M=C_out, K=C_in, N=4*P_BLOCKS
    n = k * P_BLOCKS + p  (fast index)
    """
    pid_m = tl.program_id(0)  # tile over C_out
    pid_n = tl.program_id(1)  # tile over N_out = 4 * P_BLOCKS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode spatial coordinates from linear index n = k * P_BLOCKS + p
    k_idx   = offs_n // P_BLOCKS          # kernel position: 0..3
    p_idx   = offs_n %  P_BLOCKS          # block index:     0..P_BLOCKS-1
    block_h = p_idx // N_BLK_W           # block row
    block_w = p_idx %  N_BLK_W           # block col
    h_idx   = block_h * 2 + k_idx // 2  # actual spatial row
    w_idx   = block_w * 2 + k_idx %  2  # actual spatial col

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k_base = tl.arange(0, BLOCK_K)

    for k_step in range(0, C_in // BLOCK_K):
        offs_k = k_step * BLOCK_K + offs_k_base

        # --- Weight tile [BLOCK_M, BLOCK_K] ---
        # weight[cout, cin, 0, 0] at cout * C_in + cin
        w_ptrs = weight_ptr + offs_m[:, None] * C_in + offs_k[None, :]
        w_mask = (offs_m[:, None] < C_out) & (offs_k[None, :] < C_in)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # --- Input tile [BLOCK_K, BLOCK_N] ---
        # input[0, cin, h, w] at cin * H * W + h * W + w
        i_ptrs = input_ptr + offs_k[:, None] * (H * W) + h_idx[None, :] * W + w_idx[None, :]
        i_mask = (offs_k[:, None] < C_in) & (offs_n[None, :] < N_out)
        i_tile = tl.load(i_ptrs, mask=i_mask, other=0.0)

        acc += tl.dot(w_tile, i_tile, allow_tf32=True)

    # Store to output[0, c, k, p] → linear: c * N_out + n
    out_ptrs = output_ptr + offs_m[:, None] * N_out + offs_n[None, :]
    out_mask = (offs_m[:, None] < C_out) & (offs_n[None, :] < N_out)
    tl.store(out_ptrs, acc.to(w_tile.dtype), mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_unfold(weight, input_):
    """
    Fuses:
      conv2d  = torch.conv2d(input_, weight, None, (1,1), (0,0), (1,1), 1)
      unfolded = F.unfold(conv2d, kernel_size=(2,2), stride=(2,2))
      result   = unfolded.reshape(1, C_out, 4, -1)
    """
    C_out = weight.shape[0]
    C_in  = weight.shape[1]
    H     = input_.shape[2]
    W     = input_.shape[3]

    k_size  = 2
    n_h     = H // k_size           # 16
    n_w     = W // k_size           # 16
    P_BLOCKS = n_h * n_w            # 256
    N_BLK_W  = n_w                  # 16
    N_out    = k_size * k_size * P_BLOCKS  # 1024

    output = torch.empty(
        (1, C_out, k_size * k_size, P_BLOCKS),
        dtype=input_.dtype,
        device=input_.device,
    )

    grid = lambda META: (
        triton.cdiv(C_out, META['BLOCK_M']),
        triton.cdiv(N_out,  META['BLOCK_N']),
    )

    fused_conv1x1_unfold_kernel[grid](
        input_,
        weight,
        output,
        C_out, C_in, H, W, P_BLOCKS, N_BLK_W, N_out,
    )

    return output


# -----------------------------------------------------------------------
# Pattern / replacement API
# -----------------------------------------------------------------------

def pattern(in_0, in_1):
    """
    Matches the three-op chain in model.py:
      conv2d   = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
      tmp_2    = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
      tmp_3    = tmp_2.reshape(1, 128, 4, -1)
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2  = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    tmp_3  = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_conv1x1_unfold