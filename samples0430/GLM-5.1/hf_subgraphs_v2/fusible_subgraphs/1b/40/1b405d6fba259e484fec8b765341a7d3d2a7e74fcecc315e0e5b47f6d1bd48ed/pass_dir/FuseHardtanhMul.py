import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512}, num_warps=8),
    ],
    key=['B', 'C_out', 'H', 'W'],
)
@triton.jit
def conv1x1_hardtanh_mul_kernel(
    input_ptr, weight_ptr, bias_ptr, act_ptr, output_ptr,
    B, C_in: tl.constexpr, C_out, H, W,
    s_in_b, s_in_c, s_in_h, s_in_w,
    s_wt_co, s_wt_ci,
    s_act_b, s_act_c, s_act_h, s_act_w,
    s_out_b, s_out_c, s_out_h, s_out_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = off_m < C_out
    mask_n = off_n < B * H * W

    HW = H * W
    b_idx = off_n // HW
    hw_idx = off_n % HW
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    # Accumulator for conv output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matmul loop over input channels
    for k_start in range(0, C_in, BLOCK_K):
        off_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = off_k < C_in

        # Load weight tile [BLOCK_M, BLOCK_K]
        w_ptrs = weight_ptr + off_m[:, None] * s_wt_co + off_k[None, :] * s_wt_ci
        w_mask = mask_m[:, None] & mask_k[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Load input tile [BLOCK_K, BLOCK_N]
        i_ptrs = input_ptr + b_idx[None, :] * s_in_b + off_k[:, None] * s_in_c + h_idx[None, :] * s_in_h + w_idx[None, :] * s_in_w
        i_mask = mask_k[:, None] & mask_n[None, :]
        i = tl.load(i_ptrs, mask=i_mask, other=0.0)

        # Accumulate matmul
        acc += tl.dot(w, i)

    # Add bias
    b_ptrs = bias_ptr + off_m
    b_vals = tl.load(b_ptrs, mask=mask_m, other=0.0)
    acc += b_vals[:, None]

    # Load activation (in_3) tile [BLOCK_M, BLOCK_N]
    a_ptrs = act_ptr + b_idx[None, :] * s_act_b + off_m[:, None] * s_act_c + h_idx[None, :] * s_act_h + w_idx[None, :] * s_act_w
    a_mask = mask_m[:, None] & mask_n[None, :]
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # Hardtanh: clamp to [0, 6]
    a = tl.where(a < 0.0, 0.0, a)
    a = tl.where(a > 6.0, 6.0, a)

    # Multiply conv output with hardtanh result
    result = a * acc

    # Store output
    o_ptrs = output_ptr + b_idx[None, :] * s_out_b + off_m[:, None] * s_out_c + h_idx[None, :] * s_out_h + w_idx[None, :] * s_out_w
    o_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, result, mask=o_mask)


@torch.fx.wrap
def conv1x1_hardtanh_mul_fused(bias, weight, input_tensor, activation):
    device = input_tensor.device
    if bias.device != device:
        bias = torch.as_tensor(bias, device=device)
    if weight.device != device:
        weight = torch.as_tensor(weight, device=device)

    B, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]

    s_in_b, s_in_c, s_in_h, s_in_w = input_tensor.stride()
    s_wt_co, s_wt_ci = weight.stride()[0], weight.stride()[1]
    s_act_b, s_act_c, s_act_h, s_act_w = activation.stride()

    output = torch.empty(B, C_out, H, W, dtype=input_tensor.dtype, device=device)
    s_out_b, s_out_c, s_out_h, s_out_w = output.stride()

    BLOCK_K = 32

    grid = lambda meta: (
        triton.cdiv(C_out, meta['BLOCK_M']),
        triton.cdiv(B * H * W, meta['BLOCK_N']),
    )

    conv1x1_hardtanh_mul_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        act_ptr=activation,
        output_ptr=output,
        B=B, C_in=C_in, C_out=C_out, H=H, W=W,
        s_in_b=s_in_b, s_in_c=s_in_c, s_in_h=s_in_h, s_in_w=s_in_w,
        s_wt_co=s_wt_co, s_wt_ci=s_wt_ci,
        s_act_b=s_act_b, s_act_c=s_act_c, s_act_h=s_act_h, s_act_w=s_act_w,
        s_out_b=s_out_b, s_out_c=s_out_c, s_out_h=s_out_h, s_out_w=s_out_w,
        BLOCK_K=BLOCK_K,
    )

    return output


def replacement_func():
    return conv1x1_hardtanh_mul_fused