import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 512}, num_stages=3, num_warps=8),
    ],
    key=['total_m', 'C', 'HW'],
)
@triton.jit
def _sigmoid_nchw2nhwc_kernel(
    input_ptr,
    output_ptr,
    total_m,
    C,
    HW,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Fused NCHW -> NHWC transpose + sigmoid.
    input:  [total_m // HW, C, HW]  (NCHW flattened to [N, C, HW])
    output: [total_m, C]             (NHWC flattened to [M, C])
    """
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    c_offs = tl.arange(0, BLOCK_C)                    # [BLOCK_C]

    mask_m = m_offs < total_m
    mask_c = c_offs < C
    mask = mask_m[:, None] & mask_c[None, :]           # [BLOCK_M, BLOCK_C]

    # Compute per-element batch and spatial indices
    n  = m_offs // HW   # batch index  [BLOCK_M]
    hw = m_offs  % HW   # spatial index [BLOCK_M]

    # NCHW read: input[n, c, hw]  = input_ptr + n*C*HW + c*HW + hw
    read_offs = n[:, None] * (C * HW) + c_offs[None, :] * HW + hw[:, None]
    vals = tl.load(input_ptr + read_offs, mask=mask, other=0.0)

    # Sigmoid in fp32 for numerical accuracy
    out = tl.sigmoid(vals.to(tl.float32)).to(vals.dtype)

    # NHWC write: output[m, c] = output_ptr + m*C + c
    write_offs = m_offs[:, None] * C + c_offs[None, :]
    tl.store(output_ptr + write_offs, out, mask=mask)


def sigmoid_nhwc(x, N_reshape, C_out):
    """
    x         : conv2d output, shape [N, C_out, H, W], any dtype
    N_reshape : first dim of the final reshape  (== N in all our graphs)
    C_out     : number of output channels (last dim of reshape)
    Returns   : [N_reshape, H*W*(N//N_reshape), C_out]  with sigmoid applied
    """
    N, C, H, W = x.shape
    HW       = H * W
    total_m  = N * HW

    # Allocate output in NHWC layout: [total_m, C]
    output = torch.empty((total_m, C), dtype=x.dtype, device=x.device)

    # BLOCK_C must be a compile-time constant >= C; use next power-of-2
    if C <= 1:
        BLOCK_C = 1
    elif C <= 2:
        BLOCK_C = 2
    elif C <= 4:
        BLOCK_C = 4
    elif C <= 8:
        BLOCK_C = 8
    elif C <= 16:
        BLOCK_C = 16
    elif C <= 32:
        BLOCK_C = 32
    else:
        BLOCK_C = 64

    # Launch with autotuned BLOCK_M; grid is 1-D over M
    _sigmoid_nchw2nhwc_kernel[
        lambda meta: ((total_m + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
    ](
        x, output,
        total_m, C, HW,
        BLOCK_C=BLOCK_C,
    )

    return output.reshape(N_reshape, -1, C_out)