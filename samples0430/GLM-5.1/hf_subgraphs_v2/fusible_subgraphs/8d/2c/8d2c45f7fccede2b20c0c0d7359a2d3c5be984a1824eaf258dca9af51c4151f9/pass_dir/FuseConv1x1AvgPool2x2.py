import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    # in_0 = weight, in_1 = input
    # Our kernel wrapper takes (input, weight)
    return (in_1, in_0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_CO': 64, 'BLOCK_PW': 32, 'BLOCK_CI': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_CO': 32, 'BLOCK_PW': 32, 'BLOCK_CI': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_CO': 64, 'BLOCK_PW': 32, 'BLOCK_CI': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_CO': 32, 'BLOCK_PW': 32, 'BLOCK_CI': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_CO': 128, 'BLOCK_PW': 32, 'BLOCK_CI': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_CO': 128, 'BLOCK_PW': 32, 'BLOCK_CI': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_CO': 64, 'BLOCK_PW': 16, 'BLOCK_CI': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_CO': 32, 'BLOCK_PW': 16, 'BLOCK_CI': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_CO': 128, 'BLOCK_PW': 16, 'BLOCK_CI': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_CO': 64, 'BLOCK_PW': 64, 'BLOCK_CI': 32}, num_warps=8, num_stages=3),
    ],
    key=['C_in', 'C_out', 'W_out'],
)
@triton.jit
def fused_conv1x1_avgpool2x2_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, C_in, C_out, H, W, H_out, W_out,
    stride_ib, stride_ic, stride_ih, stride_iw,
    stride_w0, stride_w1,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_CO: tl.constexpr, BLOCK_PW: tl.constexpr, BLOCK_CI: tl.constexpr,
):
    # Grid: pid_co covers output channels, pid_row covers (batch, pooled_height, pw_block)
    pid_co = tl.program_id(0)
    pid_row = tl.program_id(1)

    # Compute pw_blocks at runtime (BLOCK_PW is constexpr)
    pw_blocks = (W_out + BLOCK_PW - 1) // BLOCK_PW

    # Decode pid_row to (b, ph, pw_start)
    bph = pid_row // pw_blocks
    pid_pw = pid_row % pw_blocks
    b = bph // H_out
    ph = bph % H_out
    pw_start = pid_pw * BLOCK_PW

    # Output channel offsets
    co_start = pid_co * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_out

    # Pooled width offsets (contiguous in memory for coalescing)
    pw_offsets = pw_start + tl.arange(0, BLOCK_PW)
    pw_mask = pw_offsets < W_out

    # Input spatial positions for 2x2 avg pooling
    ih0 = 2 * ph
    ih1 = ih0 + 1
    iw0 = 2 * pw_offsets      # contiguous along W dimension
    iw1 = iw0 + 1

    # Accumulator in float32
    acc = tl.zeros((BLOCK_CO, BLOCK_PW), dtype=tl.float32)

    # Base offset for this (b, ih0/ih1) - shared across all CI iterations
    b_offset = b * stride_ib
    ih0_offset = ih0 * stride_ih
    ih1_offset = ih1 * stride_ih

    for ci_start in range(0, C_in, BLOCK_CI):
        ci_offsets = ci_start + tl.arange(0, BLOCK_CI)
        ci_mask = ci_offsets < C_in

        # Load weight block in original dtype (for tensor cores)
        w_ptrs = weight_ptr + co_offsets[:, None] * stride_w0 + ci_offsets[None, :] * stride_w1
        w = tl.load(w_ptrs, mask=co_mask[:, None] & ci_mask[None, :], other=0.0)

        # Compute base input pointer for this ci range
        ci_base = input_ptr + b_offset + ci_offsets[:, None] * stride_ic

        # Load 4 spatial positions in original dtype
        i_00 = tl.load(ci_base + ih0_offset + iw0[None, :] * stride_iw,
                       mask=ci_mask[:, None] & pw_mask[None, :], other=0.0)
        i_10 = tl.load(ci_base + ih1_offset + iw0[None, :] * stride_iw,
                       mask=ci_mask[:, None] & pw_mask[None, :], other=0.0)
        i_01 = tl.load(ci_base + ih0_offset + iw1[None, :] * stride_iw,
                       mask=ci_mask[:, None] & pw_mask[None, :], other=0.0)
        i_11 = tl.load(ci_base + ih1_offset + iw1[None, :] * stride_iw,
                       mask=ci_mask[:, None] & pw_mask[None, :], other=0.0)

        # Compute average in original dtype - enables tensor cores for float16/bfloat16
        avg = (i_00 + i_10 + i_01 + i_11) * 0.25

        # tl.dot: [BLOCK_CO, BLOCK_CI] @ [BLOCK_CI, BLOCK_PW] = [BLOCK_CO, BLOCK_PW]
        # For float16/bfloat16: uses tensor cores, result is float32
        acc += tl.dot(w, avg)

    # Store output[b, co, ph, pw]
    o_ptrs = output_ptr + b * stride_ob + co_offsets[:, None] * stride_oc + ph * stride_oh + pw_offsets[None, :] * stride_ow
    o_mask = co_mask[:, None] & pw_mask[None, :]
    tl.store(o_ptrs, acc, mask=o_mask)


@torch.fx.wrap
def fused_conv1x1_avgpool2x2(input_tensor, weight_tensor):
    B, C_in, H, W = input_tensor.shape
    C_out = weight_tensor.shape[0]
    H_out = H // 2
    W_out = W // 2

    output = torch.empty((B, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)

    stride_ib, stride_ic, stride_ih, stride_iw = input_tensor.stride()
    stride_w0 = weight_tensor.stride()[0]
    stride_w1 = weight_tensor.stride()[1]
    stride_ob, stride_oc, stride_oh, stride_ow = output.stride()

    # Grid function adapts to autotune config, includes pw tiling
    def grid(meta):
        pw_blocks = triton.cdiv(W_out, meta['BLOCK_PW'])
        return (
            triton.cdiv(C_out, meta['BLOCK_CO']),
            B * H_out * pw_blocks,
        )

    fused_conv1x1_avgpool2x2_kernel[grid](
        input_tensor, weight_tensor, output,
        B, C_in, C_out, H, W, H_out, W_out,
        stride_ib, stride_ic, stride_ih, stride_iw,
        stride_w0, stride_w1,
        stride_ob, stride_oc, stride_oh, stride_ow,
    )

    return output


def replacement_func():
    return fused_conv1x1_avgpool2x2