import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['HW'],
)
@triton.jit
def fused_linear_sigmoid_mul_kernel(
    bias_ptr, weight_ptr, input_ptr, feat_ptr, out_ptr,
    B, C: tl.constexpr, K: tl.constexpr, HW,
    stride_feat_b, stride_feat_c,
    stride_out_b, stride_out_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b = pid_bc // C
    c = pid_bc % C

    # Compute linear: dot(input[b, :], weight[c, :]) + bias[c]
    acc = tl.zeros((), dtype=tl.float32)
    input_base = b * K
    weight_base = c * K
    for k in tl.static_range(K):
        x_val = tl.load(input_ptr + input_base + k)
        w_val = tl.load(weight_ptr + weight_base + k)
        acc += x_val.to(tl.float32) * w_val.to(tl.float32)

    bias_val = tl.load(bias_ptr + c)
    acc += bias_val.to(tl.float32)

    # Sigmoid
    sig = tl.sigmoid(acc)

    # Multiply with spatial features
    hw_offset = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = hw_offset < HW

    feat_offset = b * stride_feat_b + c * stride_feat_c + hw_offset
    feat_val = tl.load(feat_ptr + feat_offset, mask=mask, other=0.0)

    out_val = feat_val * sig.to(feat_val.dtype)

    out_offset = b * stride_out_b + c * stride_out_c + hw_offset
    tl.store(out_ptr + out_offset, out_val, mask=mask)


@torch.fx.wrap
def fused_linear_sigmoid_mul(in_0, in_1, in_2, in_3):
    B = in_3.shape[0]
    C = in_3.shape[1]
    H = in_3.shape[2]
    W = in_3.shape[3]
    K = in_2.shape[1]
    HW = H * W

    out = torch.empty_like(in_3)

    grid = (B * C, (HW + 4095) // 4096)

    fused_linear_sigmoid_mul_kernel[grid](
        in_0, in_1, in_2, in_3, out,
        B, C, K, HW,
        in_3.stride(0), in_3.stride(1),
        out.stride(0), out.stride(1),
    )

    return out