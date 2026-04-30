import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64,  'BLOCK_C': 1},   num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 1},   num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 1},   num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 1},   num_warps=8),
        triton.Config({'BLOCK_HW': 1024,'BLOCK_C': 1},   num_warps=8),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 4},   num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 4},   num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 4},   num_warps=8),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 8},   num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 8},   num_warps=8),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 16},  num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 16},  num_warps=8),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 32},  num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 32},  num_warps=8),
    ],
    key=['N_HW', 'C'],
)
@triton.jit
def permute_reshape_sigmoid_kernel(
    in_ptr,
    out_ptr,
    N_HW,       # N * H * W  (total spatial elements per channel)
    C,          # number of channels
    HW,         # H * W  (spatial size of one sample)
    BLOCK_HW: tl.constexpr,
    BLOCK_C:  tl.constexpr,
):
    """
    Fused permute(0,2,3,1) + reshape(B,-1,C) + sigmoid.

    Input  layout: NCHW  (x[n, c, hw] contiguous at n*C*N_HW + c*N_HW + hw)
    Output layout: NHWC  (out[n, hw, c] contiguous at n*HW*C + hw*C + c)

    Each program handles a tile of shape [BLOCK_HW, BLOCK_C]:
      - BLOCK_HW consecutive spatial positions (hw)
      - BLOCK_C consecutive channels (c)
    """
    pid_hw = tl.program_id(0)   # block over spatial dimension
    pid_c  = tl.program_id(1)   # block over channel dimension

    hw_start = pid_hw * BLOCK_HW
    c_start  = pid_c  * BLOCK_C

    hw_offs = hw_start + tl.arange(0, BLOCK_HW)   # [BLOCK_HW]
    c_offs  = c_start  + tl.arange(0, BLOCK_C)    # [BLOCK_C]

    hw_mask = hw_offs < N_HW
    c_mask  = c_offs  < C
    mask    = hw_mask[:, None] & c_mask[None, :]   # [BLOCK_HW, BLOCK_C]

    # Global batch index from the spatial position
    # (all hw positions in this block share the same batch element)
    n_g = hw_offs // HW   # [BLOCK_HW]

    # NCHW flat indices: [n_g, c_offs, hw_offs]
    in_idx = (n_g * C * N_HW
              + c_offs[None, :] * N_HW
              + hw_offs[:, None])               # [BLOCK_HW, BLOCK_C]

    # NHWC flat indices: [n_g, hw_offs, c_offs]
    out_idx = (n_g * HW * C
               + hw_offs[:, None] * C
               + c_offs[None, :])               # [BLOCK_HW, BLOCK_C]

    x = tl.load(in_ptr + in_idx, mask=mask, other=0.0)

    # Sigmoid in fp32 for accuracy, then cast back to input dtype
    y = x.to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-y))
    y = y.to(x.dtype)

    tl.store(out_ptr + out_idx, y, mask=mask)


@torch.fx.wrap
def permute_reshape_sigmoid(x):
    """
    Fused replacement for:
        x.permute(0, 2, 3, 1)      # NCHW → NHWC view
        .reshape(N, -1, C)         # (same memory, just reinterprets strides)
        .sigmoid()                  # element-wise

    x   : [N, C, H, W]  NCHW, contiguous, any float dtype
    out : [N, H*W, C]   contiguous NHWC, same dtype
    """
    N    = x.shape[0]
    C    = x.shape[1]
    HW   = x.shape[2] * x.shape[3]
    N_HW = N * HW

    out = torch.empty((N, N_HW, C), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(N_HW, META['BLOCK_HW']),
        triton.cdiv(C,    META['BLOCK_C']),
    )

    permute_reshape_sigmoid_kernel[grid](x, out, N_HW, C, HW)

    return out