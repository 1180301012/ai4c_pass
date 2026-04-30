import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_2.softmax(dim=-1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    return (tmp_8,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_gated_softmax_kernel(
    gate_ptr,
    patch_ptr,
    pos_ptr,
    out_ptr,
    N_H,
    N_I,
    N_J,
    stride_gate,
    stride_patch_h, stride_patch_i, stride_patch_j,
    stride_pos_h, stride_pos_i, stride_pos_j,
    stride_out_h, stride_out_i, stride_out_j,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid // N_I
    i = pid % N_I

    # Load gating parameter and compute sigmoid (only once, reuse for entire row)
    gate_val = tl.load(gate_ptr + h * stride_gate).to(tl.float32)
    sig = tl.sigmoid(gate_val)
    one_minus_sig = 1.0 - sig

    # Compute offsets for j dimension
    j_offsets = tl.arange(0, BLOCK_J)
    j_mask = j_offsets < N_J

    # Load pos_score row for softmax computation
    row_base_pos = h * stride_pos_h + i * stride_pos_i
    pos_row = tl.load(pos_ptr + row_base_pos + j_offsets * stride_pos_j, mask=j_mask, other=float('-inf')).to(tl.float32)

    # Compute softmax along j dimension (dim=-1) - numerically stable
    row_max = tl.max(pos_row, axis=0)
    exp_row = tl.exp(pos_row - row_max)
    row_sum = tl.sum(exp_row, axis=0)
    softmax_row = exp_row / row_sum

    # Load patch_score row
    row_base_patch = h * stride_patch_h + i * stride_patch_i
    patch_row = tl.load(patch_ptr + row_base_patch + j_offsets * stride_patch_j, mask=j_mask, other=0.0).to(tl.float32)

    # Compute output: (1 - sigmoid(gate)) * patch_score + sigmoid(gate) * softmax(pos_score)
    out_row = one_minus_sig * patch_row + sig * softmax_row

    # Store output
    row_base_out = h * stride_out_h + i * stride_out_i
    tl.store(out_ptr + row_base_out + j_offsets * stride_out_j, out_row, mask=j_mask)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    # Move in_0 to GPU if needed (it's originally on CPU)
    if in_0.device != in_1.device:
        in_0 = in_0.to(in_1.device)

    # Ensure contiguous memory layout for Triton kernel
    in_1 = in_1.contiguous()
    in_2 = in_2.contiguous()

    N_H = in_0.shape[0]
    N_I = in_1.shape[2]
    N_J = in_1.shape[3]

    BLOCK_J = triton.next_power_of_2(N_J)

    # Create output tensor with same shape and dtype as in_1
    out = torch.empty_like(in_1)

    # Grid: one program per row (h, i) pair
    grid = (N_H * N_I,)

    fused_gated_softmax_kernel[grid](
        in_0, in_1, in_2, out,
        N_H, N_I, N_J,
        stride_gate=in_0.stride(0),
        stride_patch_h=in_1.stride(1), stride_patch_i=in_1.stride(2), stride_patch_j=in_1.stride(3),
        stride_pos_h=in_2.stride(1), stride_pos_i=in_2.stride(2), stride_pos_j=in_2.stride(3),
        stride_out_h=out.stride(1), stride_out_i=out.stride(2), stride_out_j=out.stride(3),
        BLOCK_J=BLOCK_J,
    )

    return (out,)


def replacement_func():
    return kernel_wrapper