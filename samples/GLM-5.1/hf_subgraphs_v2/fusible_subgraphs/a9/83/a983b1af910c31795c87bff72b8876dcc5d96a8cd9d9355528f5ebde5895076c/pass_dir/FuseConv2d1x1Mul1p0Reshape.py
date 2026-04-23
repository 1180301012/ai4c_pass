import torch
import triton
import triton.language as tl

def pattern(bias, weight, input):
    conv2d = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4

def replacement_args(bias, weight, input):
    return (bias, weight, input)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['total_hw', 'C_in', 'HW'],
)
@triton.jit
def conv1x1_fused_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    total_hw, C_out, C_in, HW, W_dim,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_wt_c, stride_wt_k,
    stride_out_n, stride_out_c, stride_out_hw,
    stride_bias,
    BLOCK_HW: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_HW
    offs_hw = block_start + tl.arange(0, BLOCK_HW)
    hw_mask = offs_hw < total_hw

    n_idx = offs_hw // HW
    spatial_idx = offs_hw % HW
    h_idx = spatial_idx // W_dim
    w_idx = spatial_idx % W_dim

    # Use power-of-2 (32) for channel dimension since tl.dot requires it
    C_PAD: tl.constexpr = 32
    offs_c = tl.arange(0, C_PAD)
    c_mask = offs_c < C_out
    bias_vals = tl.load(bias_ptr + offs_c * stride_bias, mask=c_mask, other=0.0)

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_HW, C_PAD), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(C_in, BLOCK_K)):
        k_offs = k_start * BLOCK_K + offs_k
        # Load weight: (C_PAD, BLOCK_K) padded from (C_out, C_in)
        weight_block = tl.load(
            weight_ptr + offs_c[:, None] * stride_wt_c + k_offs[None, :] * stride_wt_k,
            mask=c_mask[:, None] & (k_offs[None, :] < C_in),
            other=0.0
        )
        input_block = tl.load(
            input_ptr + n_idx[:, None] * stride_in_n + k_offs[None, :] * stride_in_c + spatial_idx[:, None] * stride_in_w,
            mask=hw_mask[:, None] & (k_offs[None, :] < C_in),
            other=0.0
        )
        acc += tl.dot(input_block, weight_block.T, allow_tf32=True)

    acc = acc * 1.0
    acc += bias_vals[None, :]
    out = acc.to(output_ptr.dtype.element_ty)

    tl.store(
        output_ptr + n_idx[:, None] * stride_out_n + offs_c[None, :] * stride_out_c + spatial_idx[:, None] * stride_out_hw,
        out,
        mask=hw_mask[:, None] & c_mask[None, :]
    )


@torch.fx.wrap
def conv1x1_mul_reshape(bias, weight, input):
    C_out = weight.shape[0]
    C_in = weight.shape[1]
    N_batch = input.shape[0]
    H = input.shape[2]
    W = input.shape[3]
    HW = H * W
    total_hw = N_batch * HW

    output = torch.empty(N_batch, C_out, HW, dtype=input.dtype, device=input.device)

    grid = lambda META: (triton.cdiv(total_hw, META['BLOCK_HW']),)

    conv1x1_fused_kernel[grid](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
        total_hw=total_hw, C_out=C_out, C_in=C_in, HW=HW, W_dim=W,
        stride_in_n=input.stride(0), stride_in_c=input.stride(1),
        stride_in_h=input.stride(2), stride_in_w=input.stride(3),
        stride_wt_c=weight.stride(0), stride_wt_k=weight.stride(1),
        stride_out_n=output.stride(0), stride_out_c=output.stride(1), stride_out_hw=output.stride(2),
        stride_bias=bias.stride(0),
    )

    return output

def replacement_func():
    return conv1x1_mul_reshape