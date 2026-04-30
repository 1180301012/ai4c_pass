import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match: conv2d(in_2, in_1, in_0, stride=(1,1), pad=(0,0), dil=(1,1), groups=1)
           * 1.0  ->  reshape(-1, 17, 4096)
    in_0: bias  [C_out=17]
    in_1: weight [C_out=17, C_in=256, 1, 1]
    in_2: input  [B, C_in=256, H=64, W=64]
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        # BLOCK_N must be >= C_out=17; use 32 (>=16 for tensor cores, single N-pass)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['B', 'C_in', 'C_out', 'HW'],
)
@triton.jit
def conv1x1_fused_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_in, C_out, HW,
    stride_ib, stride_ic, stride_ihw,
    stride_ob, stride_on, stride_ohw,
    BLOCK_M: tl.constexpr,   # tile over HW (spatial, K_d in matmul)
    BLOCK_N: tl.constexpr,   # tile over C_out (N dimension, >= 17)
    BLOCK_K: tl.constexpr,   # tile over C_in  (reduction dimension)
):
    # pid_bm -> (batch b, hw-tile m)
    pid_bm = tl.program_id(0)
    # Single N-tile since BLOCK_N=32 >= C_out=17
    pid_n  = tl.program_id(1)

    hw_start = pid_bm * BLOCK_M
    n_start  = pid_n  * BLOCK_N

    hw_offs = hw_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs  = n_start  + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Accumulate in float32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    hw_mask = hw_offs < HW          # [BLOCK_M]
    n_mask  = n_offs  < C_out       # [BLOCK_N]

    # Loop over the C_in reduction dimension
    num_k_blocks = tl.cdiv(C_in, BLOCK_K)
    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K
        k_offs  = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]
        k_mask  = k_offs < C_in

        # ---- Load input tile [BLOCK_M, BLOCK_K] ----
        # input[b, c, hw] at b*stride_ib + c*stride_ic + hw*stride_ihw
        # Load input[b, k_offs[j], hw_offs[i]] -> ptr[i, j]
        in_ptrs = (input_ptr
                   + (pid_bm // (HW // BLOCK_M)) * stride_ib
                   + k_offs[None, :] * stride_ic
                   + hw_offs[:, None] * stride_ihw)
        # [BLOCK_M, BLOCK_K]
        in_mask = hw_mask[:, None] & k_mask[None, :]
        in_block = tl.load(in_ptrs, mask=in_mask, other=0.0).to(tl.float32)

        # ---- Load weight tile [BLOCK_N, BLOCK_K] ----
        # weight[c_out, c_in], strides (C_in, 1)
        # weight[n, k] = weight_ptr + n * C_in + k
        wt_ptrs = weight_ptr + n_offs[:, None] * C_in + k_offs[None, :]
        # [BLOCK_N, BLOCK_K]
        wt_mask = n_mask[:, None] & k_mask[None, :]
        wt_block = tl.load(wt_ptrs, mask=wt_mask, other=0.0).to(tl.float32)

        # in_block [BLOCK_M, BLOCK_K] @ wt_block.T [BLOCK_K, BLOCK_N] => [BLOCK_M, BLOCK_N]
        acc = tl.dot(in_block, tl.trans(wt_block), acc, out_dtype=tl.float32)

    # Add bias: bias[n_offs], broadcast over hw
    bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store output[b, n, hw] with shape [B, C_out, HW]
    out_ptrs = (output_ptr
                + pid_bm  // (HW // BLOCK_M) * stride_ob
                + n_offs[None, :] * stride_on
                + hw_offs[:, None] * stride_ohw)
    out_mask = hw_mask[:, None] & n_mask[None, :]
    # Cast back to the original element dtype
    tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def conv1x1_fused(bias, weight, inp):
    """
    Fused 1x1-conv + * 1.0 + reshape for:
      bias   : [C_out=17]
      weight : [C_out=17, C_in=256, 1, 1]
      inp    : [B, C_in=256, H=64, W=64]
    Returns  : [B, C_out=17, H*W=4096]
    """
    B, C_in, H, W = inp.shape
    C_out = weight.shape[0]   # 17
    HW    = H * W             # 4096

    output = torch.empty((B, C_out, HW), dtype=inp.dtype, device=inp.device)

    # Grid: (num_hw_tiles * B, num_n_tiles)
    # BLOCK_N >= C_out so num_n_tiles = 1 always
    grid = lambda meta: (
        B * triton.cdiv(HW, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv1x1_fused_kernel[grid](
        inp, weight, bias, output,
        B, C_in, C_out, HW,
        inp.stride(0), inp.stride(1), inp.stride(2),   # stride_ib, stride_ic, stride_ihw
        output.stride(0), output.stride(1), output.stride(2),  # stride_ob, stride_on, stride_ohw
    )

    return output


def replacement_func():
    return conv1x1_fused