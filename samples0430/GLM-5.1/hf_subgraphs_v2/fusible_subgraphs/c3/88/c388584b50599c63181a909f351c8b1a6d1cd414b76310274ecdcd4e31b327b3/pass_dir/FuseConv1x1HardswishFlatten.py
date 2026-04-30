import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_conv1x1_hardswish_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, in_ch, out_ch, hw, h_dim, w_dim,
    is0, is1, is2, is3,
    ws0, ws1,
    bs0,
    os0, os1, os2, os3,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(batch * hw, BLOCK_M)
    num_pid_n = tl.cdiv(out_ch, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_off < batch * hw
    n_mask = n_off < out_ch

    # Decode m to (batch_idx, h_idx, w_idx)
    b_idx = m_off // hw
    hw_idx = m_off % hw
    h_idx = hw_idx // w_dim
    w_idx = hw_idx % w_dim

    # Base offsets for input [N, CI, H, W] and output [N, CO, H, W]
    in_base = b_idx * is0 + h_idx * is2 + w_idx * is3
    out_base = b_idx * os0 + h_idx * os2 + w_idx * os3

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Reduction loop over input channels
    for k_start in range(0, in_ch, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < in_ch

        # Load input tile [BLOCK_M, BLOCK_K]
        # input[b_idx, k_off, h_idx, w_idx]
        in_offsets = in_base[:, None] + k_off[None, :] * is1
        in_load_mask = m_mask[:, None] & k_mask[None, :]
        x = tl.load(input_ptr + in_offsets, mask=in_load_mask, other=0.0)

        # Load weight tile [BLOCK_K, BLOCK_N] (for tl.dot)
        # weight[k_off, n_off] treating weight as [CO, CI] matrix
        wt_offsets = k_off[:, None] * ws1 + n_off[None, :] * ws0
        wt_load_mask = k_mask[:, None] & n_mask[None, :]
        w = tl.load(weight_ptr + wt_offsets, mask=wt_load_mask, other=0.0)

        # Matrix multiply: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] = [BLOCK_M, BLOCK_N]
        acc += tl.dot(x, w, allow_tf32=True)

    # Add bias [BLOCK_N]
    b = tl.load(bias_ptr + n_off * bs0, mask=n_mask, other=0.0).to(tl.float32)
    acc += b[None, :]

    # HardSwish activation: hardswish(x) = x * relu6(x + 3) / 6
    # relu6(x) = min(max(x, 0), 6)
    acc = acc * tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0) / 6.0

    # Store output in [N, CO, H, W] layout
    # output[b_idx, n_off, h_idx, w_idx]
    out_offsets = out_base[:, None] + n_off[None, :] * os1
    out_store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_store_mask)


@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(bias, weight, input_tensor):
    N, CI, H, W = input_tensor.shape
    CO = weight.shape[0]
    HW = H * W

    # Create output tensor in [N, CO, H, W] layout
    output = torch.empty((N, CO, H, W), dtype=input_tensor.dtype, device=input_tensor.device)

    # Get strides
    is0, is1, is2, is3 = input_tensor.stride()
    ws0, ws1 = weight.stride()[0], weight.stride()[1]
    bs0 = bias.stride()[0]
    os0, os1, os2, os3 = output.stride()

    # Fixed block sizes - tuned for 1x1 conv shapes
    # Larger BLOCK_N for better tensor core utilization
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 32

    M = N * HW
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (CO + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m * grid_n,)

    fused_conv1x1_hardswish_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch=N,
        in_ch=CI,
        out_ch=CO,
        hw=HW,
        h_dim=H,
        w_dim=W,
        is0=is0, is1=is1, is2=is2, is3=is3,
        ws0=ws0, ws1=ws1,
        bs0=bs0,
        os0=os0, os1=os1, os2=os2, os3=os3,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_stages=4,
        num_warps=4,
    )

    # Flatten from dim 1 onwards (this is a view, no data movement)
    return output.flatten(1, -1)


def replacement_func():
    return fused_conv1x1_hardswish_flatten