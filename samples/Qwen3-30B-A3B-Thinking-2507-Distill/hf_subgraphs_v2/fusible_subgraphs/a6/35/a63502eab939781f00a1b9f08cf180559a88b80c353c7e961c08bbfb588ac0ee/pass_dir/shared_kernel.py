import torch
import triton
import triton.language as tl


@triton.jit
def fused_view_roll_ln_add_kernel(
    in0_ptr,           # bias  [C]
    in1_ptr,           # weight [C]
    in2_ptr,           # residual [1, N, C]
    in3_ptr,           # input tensor ptr
    out_ptr,           # output [1, N, C]
    s3_0, s3_1, s3_2, s3_3, s3_4, s3_5,  # 6D strides of in_3
    H, W, C,
    H2,                # in_3.shape[2] — sub-blocks in H dim
    W2,                # in_3.shape[4] — sub-blocks in W dim
    IS_FP16: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one output row (one (h,w) position)
    row_id = tl.program_id(0)

    HW = H * W
    row_i  = row_id // HW
    row_hw = row_id % HW
    row_h  = row_hw // W
    row_w  = row_hw % W

    # Cyclic roll: output (h,w) pulls from input at (h-shift, w-shift)
    SHIFT = 4
    src_h = (row_h - SHIFT + H) % H
    src_w = (row_w - SHIFT + W) % W

    ch_offs = tl.arange(0, BLOCK_C)
    mask    = ch_offs < C

    # Decompose src_h -> (h1, h2) and src_w -> (w1, w2):
    #   src_h = h1 * H2 + h2,  src_w = w1 * W2 + w2
    # Then compute the correct 6D offset using the actual strides.
    # Works for both contiguous and non-contiguous (permuted) 6D tensors.
    h1 = src_h // H2
    h2 = src_h - h1 * H2
    w1 = src_w // W2
    w2 = src_w - w1 * W2

    in3_off = (row_i * s3_0
               + h1 * s3_1
               + h2 * s3_2
               + w1 * s3_3
               + w2 * s3_4
               + ch_offs * s3_5)
    x = tl.load(in3_ptr + in3_off, mask=mask, other=0.0).to(tl.float32)

    # Layer-norm: mean & variance
    x_sum  = tl.sum(x, axis=0)
    mean   = x_sum / C
    x_c    = x - mean
    var    = tl.sum(x_c * x_c, axis=0) / C
    rstd   = tl.rsqrt(var + 1e-5)
    x_norm = x_c * rstd

    # Scale & shift
    w = tl.load(in1_ptr + ch_offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(in0_ptr + ch_offs, mask=mask, other=0.0).to(tl.float32)
    x_ln = x_norm * w + b

    # Residual add
    residual = tl.load(in2_ptr + row_id * C + ch_offs, mask=mask, other=0.0).to(tl.float32)
    result   = x_ln + residual

    # Cast back to input dtype
    if IS_FP16:
        result_cast = result.to(tl.float16)
    else:
        result_cast = result.to(tl.bfloat16)

    tl.store(out_ptr + row_id * C + ch_offs, result_cast, mask=mask)


@torch.fx.wrap
def fused_view_roll_ln_add(in_0, in_1, in_2, in_3, H, W, C):
    """Fused: view + roll + view + layer_norm + residual_add."""
    s    = in_3.stride()
    H2   = in_3.shape[2]   # sub-blocks in H-dimension (from 6D shape)
    W2   = in_3.shape[4]   # sub-blocks in W-dimension (from 6D shape)

    N      = H * W
    is_fp16 = (in_2.dtype == torch.float16)
    out    = torch.empty_like(in_2)

    BLOCK_C = 1024 if C <= 1024 else 2048

    fused_view_roll_ln_add_kernel[(N,)](
        in_0, in_1, in_2, in_3, out,
        s[0], s[1], s[2], s[3], s[4], s[5],
        H, W, C,
        H2, W2,
        IS_FP16=is_fp16,
        BLOCK_C=BLOCK_C,
        num_warps=8,
    )
    return out