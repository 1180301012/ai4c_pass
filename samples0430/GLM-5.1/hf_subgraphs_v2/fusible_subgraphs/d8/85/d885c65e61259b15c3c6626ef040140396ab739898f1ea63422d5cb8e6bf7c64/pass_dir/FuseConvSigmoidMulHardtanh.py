import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def conv1x1_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_in: tl.constexpr, C_out,
    input_stride_n, input_stride_c,
    weight_stride_oc, weight_stride_ic,
    output_stride_n, output_stride_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Computes sigmoid(conv1x1(input, weight, bias)) for 1x1 conv with 1x1 spatial input."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < N
    n_mask = n_offsets < C_out

    # Accumulator in float32 for numerical stability
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)

    # Loop over K dimension (C_in) in blocks
    for k_start in range(0, tl.cdiv(C_in, BLOCK_K)):
        k_offsets = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < C_in

        # Load input[m, k] - shape [BLOCK_M, BLOCK_K]
        a = tl.load(
            input_ptr + m_offsets[:, None] * input_stride_n + k_offsets[None, :] * input_stride_c,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        )

        # Load weight in transposed layout [BLOCK_K, BLOCK_N] for tl.dot
        # weight[k, n] = weight_ptr + k * stride_ic + n * stride_oc
        b = tl.load(
            weight_ptr + k_offsets[:, None] * weight_stride_ic + n_offsets[None, :] * weight_stride_oc,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        )

        # Matrix multiply using tensor cores
        acc += tl.dot(a, b, allow_tf32=False)

    # Add bias (broadcast over M dimension)
    bias_vals = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # Apply sigmoid
    result = tl.sigmoid(acc)

    # Store output
    out_offsets = m_offsets[:, None] * output_stride_n + n_offsets[None, :] * output_stride_c
    tl.store(output_ptr + out_offsets, result, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def fused_mul_clamp_kernel(
    feature_ptr, scale_ptr, output_ptr,
    N, C_out, H, W, HW,
    feat_stride_n, feat_stride_c, feat_stride_h, feat_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_CH: tl.constexpr, BLOCK_SPATIAL: tl.constexpr,
):
    """Computes clamp(feature * scale, 0.0, 6.0) with scale broadcast from [N, C_out]."""
    pid_ch = tl.program_id(0)
    pid_sp = tl.program_id(1)

    total_channels = N * C_out
    ch_offsets = pid_ch * BLOCK_CH + tl.arange(0, BLOCK_CH)
    ch_mask = ch_offsets < total_channels

    # Decompose channel index into (n, c)
    n = ch_offsets // C_out
    c = ch_offsets - n * C_out

    sp_offsets = pid_sp * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    sp_mask = sp_offsets < HW

    # Decompose spatial index into (h, w)
    h = sp_offsets // W
    w = sp_offsets - h * W

    # Combined 2D mask
    mask_2d = ch_mask[:, None] & sp_mask[None, :]

    # Load scale values for each channel
    scale_vals = tl.load(scale_ptr + ch_offsets, mask=ch_mask, other=0.0).to(tl.float32)

    # Feature map offsets using strides
    feat_offsets = (
        n[:, None] * feat_stride_n
        + c[:, None] * feat_stride_c
        + h[None, :] * feat_stride_h
        + w[None, :] * feat_stride_w
    )
    feat_vals = tl.load(feature_ptr + feat_offsets, mask=mask_2d, other=0.0).to(tl.float32)

    # Compute: hardtanh(feature * scale, 0.0, 6.0) = clamp(feature * scale, 0.0, 6.0)
    result = feat_vals * scale_vals[:, None]
    result = tl.minimum(result, 6.0)
    result = tl.maximum(result, 0.0)

    # Output offsets using strides
    out_offsets = (
        n[:, None] * out_stride_n
        + c[:, None] * out_stride_c
        + h[None, :] * out_stride_h
        + w[None, :] * out_stride_w
    )
    tl.store(output_ptr + out_offsets, result, mask=mask_2d)


@torch.fx.wrap
def fused_conv_sigmoid_mul_hardtanh(in_0, in_1, in_2, in_3):
    # in_0: bias [C_out]
    # in_1: weight [C_out, C_in, 1, 1]
    # in_2: feature map [N, C_out, H, W]
    # in_3: conv input [N, C_in, 1, 1]

    device = in_2.device
    dtype = in_2.dtype

    # Move weight and bias to GPU with correct dtype using allowed API
    in_0 = torch.as_tensor(in_0, device=device, dtype=dtype)
    in_1 = torch.as_tensor(in_1, device=device, dtype=dtype)
    in_3 = torch.as_tensor(in_3, device=device, dtype=dtype)

    N = in_3.shape[0]
    C_in = in_3.shape[1]
    C_out = in_1.shape[0]
    H = in_2.shape[2]
    W = in_2.shape[3]
    HW = H * W

    # Step 1: Compute conv + sigmoid → scale [N, C_out]
    scale = torch.empty((N, C_out), dtype=dtype, device=device)

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32

    grid_m = triton.cdiv(N, BLOCK_M)
    grid_n = triton.cdiv(C_out, BLOCK_N)

    conv1x1_sigmoid_kernel[(grid_m, grid_n)](
        input_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=scale,
        N=N,
        C_in=C_in,
        C_out=C_out,
        input_stride_n=in_3.stride(0),
        input_stride_c=in_3.stride(1),
        weight_stride_oc=in_1.stride(0),
        weight_stride_ic=in_1.stride(1),
        output_stride_n=scale.stride(0),
        output_stride_c=scale.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Step 2: Compute broadcast multiply + hardtanh
    output = torch.empty_like(in_2)

    BLOCK_CH = 16
    BLOCK_SPATIAL = 128

    grid_ch = triton.cdiv(N * C_out, BLOCK_CH)
    grid_sp = triton.cdiv(HW, BLOCK_SPATIAL)

    fused_mul_clamp_kernel[(grid_ch, grid_sp)](
        feature_ptr=in_2,
        scale_ptr=scale,
        output_ptr=output,
        N=N,
        C_out=C_out,
        H=H,
        W=W,
        HW=HW,
        feat_stride_n=in_2.stride(0),
        feat_stride_c=in_2.stride(1),
        feat_stride_h=in_2.stride(2),
        feat_stride_w=in_2.stride(3),
        out_stride_n=output.stride(0),
        out_stride_c=output.stride(1),
        out_stride_h=output.stride(2),
        out_stride_w=output.stride(3),
        BLOCK_CH=BLOCK_CH,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )

    return (output,)


def replacement_func():
    return fused_conv_sigmoid_mul_hardtanh