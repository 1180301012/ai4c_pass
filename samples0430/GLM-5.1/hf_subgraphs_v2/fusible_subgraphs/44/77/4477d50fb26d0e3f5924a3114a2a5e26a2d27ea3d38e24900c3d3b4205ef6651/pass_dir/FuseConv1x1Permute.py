import torch
import triton
import triton.language as tl


# Pattern: 1x1 conv2d + permute(0,2,3,1)
# This matches the first two operations in all target graphs.
# The remaining reshape + sigmoid are handled by the original graph,
# but our fused output is contiguous which makes them more efficient.
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    return (tmp_3,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_conv1x1_permute_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_in, H, W, C_out,
    stride_ib, stride_ic, stride_ih, stride_iw,
    stride_wc, stride_wi,
    stride_ob, stride_oh, stride_ow, stride_oc,
    BLOCK_HW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """
    Fused 1x1 conv2d + permute kernel.
    Computes: output[b, h, w, co] = sum_ci(weight[co, ci] * input[b, ci, h, w]) + bias[co]
    Output is in contiguous [B, H, W, C_out] layout (better than permuted non-contiguous layout).
    BLOCK_CO is chosen dynamically based on C_out to minimize compute waste.
    """
    pid = tl.program_id(0)

    # Compute spatial position offsets for this program
    hw_start = pid * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)

    total_hw = B * H * W
    mask_hw = hw_offsets < total_hw

    # Decode spatial positions into (batch, height, width) indices
    b_idx = hw_offsets // (H * W)
    spatial_idx = hw_offsets % (H * W)
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W

    # Output channel offsets (BLOCK_CO covers all C_out channels)
    co_offsets = tl.arange(0, BLOCK_CO)
    mask_co = co_offsets < C_out

    # Load bias values for all C_out channels (masked for C_out < BLOCK_CO)
    bias_vals = tl.load(bias_ptr + co_offsets, mask=mask_co, other=0.0)

    # Initialize accumulator with bias (float32 for accuracy)
    acc = tl.zeros((BLOCK_HW, BLOCK_CO), dtype=tl.float32)
    # Add bias for valid positions (broadcast over HW dimension)
    acc += bias_vals[None, :].to(tl.float32)

    # Loop over input channels in blocks
    for ci_start in range(0, C_in, BLOCK_CI):
        ci_offsets = ci_start + tl.arange(0, BLOCK_CI)
        mask_ci = ci_offsets < C_in

        # Load input block: [BLOCK_HW, BLOCK_CI]
        # input[b, ci, h, w] at offset b*stride_ib + ci*stride_ic + h*stride_ih + w*stride_iw
        input_idx = (
            b_idx[:, None] * stride_ib
            + ci_offsets[None, :] * stride_ic
            + h_idx[:, None] * stride_ih
            + w_idx[:, None] * stride_iw
        )
        x = tl.load(
            input_ptr + input_idx,
            mask=mask_hw[:, None] & mask_ci[None, :],
            other=0.0,
        )

        # Load weight block: [BLOCK_CO, BLOCK_CI]
        # weight[co, ci, 0, 0] at offset co*stride_wc + ci*stride_wi
        weight_idx = co_offsets[:, None] * stride_wc + ci_offsets[None, :] * stride_wi
        w = tl.load(
            weight_ptr + weight_idx,
            mask=mask_co[:, None] & mask_ci[None, :],
            other=0.0,
        )

        # Matrix multiply: acc += x @ w.T
        # x: [BLOCK_HW, BLOCK_CI], w: [BLOCK_CO, BLOCK_CI]
        # tl.dot(x, tl.trans(w)) -> [BLOCK_HW, BLOCK_CO]
        # Note: tl.dot uses tensor cores for fp16/bf16 (produces fp32 output)
        acc += tl.dot(x, tl.trans(w))

    # Store result in contiguous [B, H, W, C_out] layout
    # output[b, h, w, co] at offset b*stride_ob + h*stride_oh + w*stride_ow + co*stride_oc
    output_idx = (
        b_idx[:, None] * stride_ob
        + h_idx[:, None] * stride_oh
        + w_idx[:, None] * stride_ow
        + co_offsets[None, :] * stride_oc
    )
    tl.store(output_ptr + output_idx, acc, mask=mask_hw[:, None] & mask_co[None, :])


def _get_block_co(c_out):
    """Select optimal BLOCK_CO based on C_out to minimize compute waste.
    Must be a multiple of 8 for tl.dot tensor core compatibility on Ampere."""
    if c_out <= 8:
        return 8
    elif c_out <= 16:
        return 16
    elif c_out <= 32:
        return 32
    else:
        return 64


@torch.fx.wrap
def fused_conv1x1_permute(bias, weight, input_tensor):
    """
    Fused 1x1 convolution + permute operation.
    Computes conv2d(input, weight, bias) with 1x1 kernel, then permutes to [B, H, W, C_out].
    Output is contiguous, which makes subsequent reshape+sigmoid more efficient.
    """
    B = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    C_out = weight.shape[0]

    # Create output in contiguous [B, H, W, C_out] layout
    output = torch.empty((B, H, W, C_out), dtype=input_tensor.dtype, device=input_tensor.device)

    # Get strides from tensors
    s_ib, s_ic, s_ih, s_iw = input_tensor.stride()
    s_wc = weight.stride()[0]  # stride for C_out dim in [C_out, C_in, 1, 1]
    s_wi = weight.stride()[1]  # stride for C_in dim
    s_ob, s_oh, s_ow, s_oc = output.stride()

    # Dynamic block sizes based on input dimensions
    BLOCK_CO = _get_block_co(C_out)

    # Adjust BLOCK_HW based on total work for better GPU utilization
    total_hw = B * H * W
    if total_hw < 8192:
        BLOCK_HW = 32  # More programs for small batch sizes
    elif total_hw < 32768:
        BLOCK_HW = 64
    else:
        BLOCK_HW = 128

    BLOCK_CI = 32

    grid = ((total_hw + BLOCK_HW - 1) // BLOCK_HW,)

    fused_conv1x1_permute_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        B=B,
        C_in=C_in,
        H=H,
        W=W,
        C_out=C_out,
        stride_ib=s_ib,
        stride_ic=s_ic,
        stride_ih=s_ih,
        stride_iw=s_iw,
        stride_wc=s_wc,
        stride_wi=s_wi,
        stride_ob=s_ob,
        stride_oh=s_oh,
        stride_ow=s_ow,
        stride_oc=s_oc,
        BLOCK_HW=BLOCK_HW,
        BLOCK_CI=BLOCK_CI,
        BLOCK_CO=BLOCK_CO,
    )

    return output


def replacement_func():
    return fused_conv1x1_permute