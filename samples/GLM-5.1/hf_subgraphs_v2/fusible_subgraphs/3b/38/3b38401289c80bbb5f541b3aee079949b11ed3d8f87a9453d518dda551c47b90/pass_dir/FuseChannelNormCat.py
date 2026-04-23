import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    return (tmp_11,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_normalize_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    stride_b_in0, stride_c_in0,
    stride_b_in1, stride_c_in1,
    stride_b_out, stride_c_out,
    n_spatial,
    LOOP_SIZE: tl.constexpr,
):
    # 2D grid: (B*3, 1) - each program handles all spatial elements for one (b, c)
    pid_bc = tl.program_id(0)
    b = pid_bc // 3
    c = pid_bc % 3

    out_base = b * stride_b_out + c * stride_c_out

    # Process spatial elements in vectorized blocks
    for start in tl.range(0, n_spatial, LOOP_SIZE):
        offsets = start + tl.arange(0, LOOP_SIZE)
        mask = offsets < n_spatial

        if c == 0:
            in_base = b * stride_b_in1
            val = tl.load(in_1_ptr + in_base + offsets, mask=mask, other=0.0).to(tl.float32)
            result = val * 0.458 - 0.030000000000000027
        elif c == 1:
            in_base = b * stride_b_in0 + 1 * stride_c_in0
            val = tl.load(in_0_ptr + in_base + offsets, mask=mask, other=0.0).to(tl.float32)
            result = val * 0.448 - 0.08799999999999997
        else:
            in_base = b * stride_b_in0 + 2 * stride_c_in0
            val = tl.load(in_0_ptr + in_base + offsets, mask=mask, other=0.0).to(tl.float32)
            result = val * 0.45 - 0.18799999999999994

        tl.store(out_ptr + out_base + offsets, result, mask=mask)


@torch.fx.wrap
def fused_normalize(in_0, in_1):
    B = in_0.shape[0]
    H = in_0.shape[2]
    W = in_0.shape[3]
    n_spatial = H * W

    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)

    LOOP_SIZE = 128  # Vector size for each loop iteration

    grid = (B * 3, 1)

    fused_normalize_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        stride_b_in0=in_0.stride(0), stride_c_in0=in_0.stride(1),
        stride_b_in1=in_1.stride(0), stride_c_in1=in_1.stride(1),
        stride_b_out=out.stride(0), stride_c_out=out.stride(1),
        n_spatial=n_spatial,
        LOOP_SIZE=LOOP_SIZE,
    )

    return out


def replacement_func():
    return fused_normalize