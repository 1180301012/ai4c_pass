import torch
import triton
import triton.language as tl


def pattern(bias, weight, residual, input_tensor):
    conv = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    drop = torch.nn.functional.dropout(conv, 0.0, False, False)
    out = drop + residual
    return (out,)


def replacement_args(bias, weight, residual, input_tensor):
    return (bias, weight, residual, input_tensor)


@triton.autotune(
    configs=[
        triton.Config({'BM': 16, 'BN': 64, 'BK': 32}, num_stages=2, num_warps=2),
        triton.Config({'BM': 16, 'BN': 128, 'BK': 32}, num_stages=2, num_warps=2),
        triton.Config({'BM': 16, 'BN': 64, 'BK': 64}, num_stages=2, num_warps=2),
        triton.Config({'BM': 16, 'BN': 128, 'BK': 64}, num_stages=2, num_warps=4),
        triton.Config({'BM': 32, 'BN': 64, 'BK': 32}, num_stages=2, num_warps=2),
        triton.Config({'BM': 32, 'BN': 128, 'BK': 32}, num_stages=2, num_warps=4),
        triton.Config({'BM': 32, 'BN': 64, 'BK': 64}, num_stages=2, num_warps=4),
        triton.Config({'BM': 32, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=4),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_stages=2, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_stages=3, num_warps=4),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 64}, num_stages=3, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
    ],
    key=['C_out', 'C_in', 'HW'],
)
@triton.jit
def fused_conv1x1_dot_kernel(
    input_ptr, weight_ptr, bias_ptr, residual_ptr, output_ptr,
    N_batch, C_out, C_in, HW,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    off_m = pid_m * BM + tl.arange(0, BM)
    off_n = pid_n * BN + tl.arange(0, BN)

    batch_offset_in = pid_b * C_in * HW
    batch_offset_out = pid_b * C_out * HW

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k_start in range(0, C_in, BK):
        off_k = k_start + tl.arange(0, BK)

        w_ptrs = weight_ptr + off_m[:, None] * C_in + off_k[None, :]
        w_mask = (off_m[:, None] < C_out) & (off_k[None, :] < C_in)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        i_ptrs = input_ptr + batch_offset_in + off_k[:, None] * HW + off_n[None, :]
        i_mask = (off_k[:, None] < C_in) & (off_n[None, :] < HW)
        i = tl.load(i_ptrs, mask=i_mask, other=0.0)

        acc += tl.dot(w, i, allow_tf32=True)

    b_ptrs = bias_ptr + off_m
    b_mask = off_m < C_out
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    acc += b[:, None]

    r_ptrs = residual_ptr + batch_offset_out + off_m[:, None] * HW + off_n[None, :]
    r_mask = (off_m[:, None] < C_out) & (off_n[None, :] < HW)
    r = tl.load(r_ptrs, mask=r_mask, other=0.0)
    acc += r

    o_ptrs = output_ptr + batch_offset_out + off_m[:, None] * HW + off_n[None, :]
    o_mask = (off_m[:, None] < C_out) & (off_n[None, :] < HW)
    tl.store(o_ptrs, acc, mask=o_mask)


@torch.fx.wrap
def fused_conv1x1_dropout0_add(bias, weight, residual, input_tensor):
    N, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    HW = H * W

    output = torch.empty_like(residual)

    grid = lambda meta: (
        triton.cdiv(C_out, meta['BM']),
        triton.cdiv(HW, meta['BN']),
        N,
    )

    fused_conv1x1_dot_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        residual_ptr=residual,
        output_ptr=output,
        N_batch=N,
        C_out=C_out,
        C_in=C_in,
        HW=HW,
    )

    return output


def replacement_func():
    return fused_conv1x1_dropout0_add