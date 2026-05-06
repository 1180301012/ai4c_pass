import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 512, 'BLOCK_HW': 16, 'BLOCK_COUT': 1, 'num_warps': 1}),
        triton.Config({'BLOCK_K': 512, 'BLOCK_HW': 32, 'BLOCK_COUT': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_K': 512, 'BLOCK_HW': 64, 'BLOCK_COUT': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_K': 512, 'BLOCK_HW': 128, 'BLOCK_COUT': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_K': 512, 'BLOCK_HW': 256, 'BLOCK_COUT': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_K': 512, 'BLOCK_HW': 512, 'BLOCK_COUT': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_K': 256, 'BLOCK_HW': 32, 'BLOCK_COUT': 4, 'num_warps': 2}),
        triton.Config({'BLOCK_K': 256, 'BLOCK_HW': 64, 'BLOCK_COUT': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_K': 256, 'BLOCK_HW': 128, 'BLOCK_COUT': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_K': 256, 'BLOCK_HW': 256, 'BLOCK_COUT': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_HW': 64, 'BLOCK_COUT': 8, 'num_warps': 4}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_HW': 128, 'BLOCK_COUT': 8, 'num_warps': 8}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_HW': 256, 'BLOCK_COUT': 8, 'num_warps': 8}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_HW': 128, 'BLOCK_COUT': 16, 'num_warps': 4}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_HW': 256, 'BLOCK_COUT': 16, 'num_warps': 8}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_HW': 256, 'BLOCK_COUT': 16, 'num_warps': 4}),
    ],
    key=['N_spatial', 'C_in', 'Cout'],
)
@triton.jit
def _fused_conv1x1_permute_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_in, Cout, N_spatial,
    BLOCK_K: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_COUT: tl.constexpr,
):
    """
    Fused: conv2d(1x1 with no padding/stride) + permute(0,2,3,1) + reshape(B,-1,Cout) + sigmoid

    input_ptr  : contiguous [B, C_in, N_spatial]  (N_spatial = H*W)
    weight_ptr : contiguous [Cout, C_in]           (1x1 conv weight flattened)
    bias_ptr   : contiguous [Cout]
    output_ptr : contiguous [B, N_spatial, Cout]
    """
    pid_batch_hw  = tl.program_id(0)   # flattens B x (N_spatial//BLOCK_HW)
    pid_cout_tile = tl.program_id(1)   # block index over Cout

    batch_idx = pid_batch_hw // (N_spatial // BLOCK_HW)
    hw_block  = pid_batch_hw %  (N_spatial // BLOCK_HW)

    hw_start  = hw_block  * BLOCK_HW
    ch_start  = pid_cout_tile * BLOCK_COUT

    hw_offs  = hw_start   + tl.arange(0, BLOCK_HW)
    out_ch   = ch_start   + tl.arange(0, BLOCK_COUT)
    hw_mask  = hw_offs   < N_spatial
    out_mask = (out_ch < Cout) & hw_mask

    acc = tl.zeros((BLOCK_HW, BLOCK_COUT), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < C_in

        # A_tile : [BLOCK_HW, BLOCK_K]
        # a_ptrs[w, k] = input[batch, k, hw]  →  input_ptr + batch*C_in*N_spatial + k*N_spatial + w
        a_ptrs = (input_ptr
                  + batch_idx * C_in * N_spatial
                  + k_offs[None, 0] * N_spatial
                  + hw_offs[0, None])
        a = tl.load(a_ptrs,
                    mask=hw_mask[None, :] & k_mask[None, :],
                    other=0.0)

        # B_tile : [BLOCK_K, BLOCK_COUT]
        # b_ptrs[k, c] = weight[out_ch+c, k]  →  weight_ptr + (out_ch+c)*C_in + k
        b_ptrs = weight_ptr + out_ch[None, :] * C_in + k_offs[0, :]
        b = tl.load(b_ptrs,
                    mask=k_mask[:, None] & (out_ch[None, :] < Cout),
                    other=0.0)

        # acc[w, c] += sum_k A[w,k] * B[k,c]
        acc = tl.dot(tl.trans(a), b, acc, out_dtype=tl.float32)

    # Add bias
    bias = tl.load(bias_ptr + out_ch, mask=out_ch < Cout, other=0.0)
    acc  = acc + bias[None, :].to(tl.float32)

    # Sigmoid
    result = tl.sigmoid(acc)

    # Store → output[batch, hw, c]
    out_ptrs = output_ptr + batch_idx * N_spatial * Cout + hw_offs[:, None] * Cout + out_ch[None, :]
    tl.store(out_ptrs, result.to(output_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_permute_reshape_sigmoid(bias, weight, x):
    """
    bias   : [Cout]
    weight : [Cout, C_in, 1, 1]
    x      : [B, C_in, H, W]  (contiguous NCHW)
    returns: [B, H*W, Cout]
    """
    B     = x.shape[0]
    C_in  = x.shape[1]
    H     = x.shape[2]
    W     = x.shape[3]
    Cout  = weight.shape[0]
    N_spatial = H * W

    output = torch.empty((B, N_spatial, Cout), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        B * (N_spatial // meta['BLOCK_HW']),
        (Cout + meta['BLOCK_COUT'] - 1) // meta['BLOCK_COUT'],
    )

    _fused_conv1x1_permute_sigmoid_kernel[grid](
        x, weight.view(Cout, C_in), bias, output,
        B, C_in, Cout, N_spatial,
    )

    return output