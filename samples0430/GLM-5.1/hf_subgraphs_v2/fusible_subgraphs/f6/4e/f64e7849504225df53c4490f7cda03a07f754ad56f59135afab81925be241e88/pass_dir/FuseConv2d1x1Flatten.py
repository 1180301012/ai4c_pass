import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return (tmp_3,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['B', 'Cin', 'Cout', 'HW'],
)
@triton.jit
def fused_1x1_conv_flatten_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, Cin, Cout, HW,
    stride_ib, stride_ic, stride_i_spatial,
    stride_wc, stride_wk,
    stride_ob, stride_oc, stride_ohw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < Cout
    n_mask = n_offsets < HW

    # Accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over input channels in tiles of BLOCK_K
    for k_start in range(0, Cin, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < Cin

        # Load weight tile [BLOCK_M, BLOCK_K]
        w_ptrs = weight_ptr + m_offsets[:, None] * stride_wc + k_offsets[None, :] * stride_wk
        w_mask = m_mask[:, None] & k_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Load input tile [BLOCK_K, BLOCK_N] using simplified 3D-like stride
        # For contiguous [B, Cin, H, W] tensors: stride_i_spatial = stride_iw = 1
        # offset for input[b, k, n] = b * stride_ib + k * stride_ic + n * stride_i_spatial
        # This avoids integer division n // W and n % W in the kernel
        i_ptrs = input_ptr + pid_b * stride_ib + k_offsets[:, None] * stride_ic + n_offsets[None, :] * stride_i_spatial
        i_mask = k_mask[:, None] & n_mask[None, :]
        i = tl.load(i_ptrs, mask=i_mask, other=0.0)

        # Matrix multiply-accumulate using tensor cores
        # allow_tf32=True enables TF32 tensor cores for fp32 inputs on Ampere GPUs
        # This gives ~8x speedup for fp32 while maintaining acceptable accuracy
        # For fp16/bf16 inputs, this flag has no effect (native tensor cores are always used)
        acc += tl.dot(w, i, allow_tf32=True)

    # Add bias (broadcast across spatial positions)
    bias_ptrs = bias_ptr + m_offsets
    bias_val = tl.load(bias_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    acc += bias_val[:, None]

    # Store output tile [BLOCK_M, BLOCK_N]
    o_ptrs = output_ptr + pid_b * stride_ob + m_offsets[:, None] * stride_oc + n_offsets[None, :] * stride_ohw
    o_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(o_ptrs, acc, mask=o_mask)


@torch.fx.wrap
def fused_1x1_conv_flatten(bias, weight, input):
    B, Cin, H, W_spatial = input.shape
    Cout = weight.shape[0]
    HW = H * W_spatial

    # Allocate output directly in flattened shape [B, Cout, HW]
    output = torch.empty((B, Cout, HW), dtype=input.dtype, device=input.device)

    # Get strides from original tensors
    # For contiguous [B, Cin, H, W] tensors, stride_i_spatial = stride_iw = 1
    # This allows us to use a simplified 3D-like access pattern: input[b, k, n] = offset with n * stride_i_spatial
    # avoiding integer division n // W and n % W in the kernel
    input_strides = input.stride()
    s_ib = input_strides[0]
    s_ic = input_strides[1]
    s_i_spatial = input_strides[3]  # stride_iw, equals 1 for contiguous tensors

    weight_strides = weight.stride()
    s_wc = weight_strides[0]
    s_wk = weight_strides[1]

    output_strides = output.stride()
    s_ob = output_strides[0]
    s_oc = output_strides[1]
    s_ohw = output_strides[2]

    # Grid dimensions depend on autotune config
    grid = lambda META: (
        (Cout + META['BLOCK_M'] - 1) // META['BLOCK_M'],
        (HW + META['BLOCK_N'] - 1) // META['BLOCK_N'],
        B,
    )

    fused_1x1_conv_flatten_kernel[grid](
        input, weight, bias, output,
        B, Cin, Cout, HW,
        s_ib, s_ic, s_i_spatial,
        s_wc, s_wk,
        s_ob, s_oc, s_ohw,
    )

    return output


def replacement_func():
    return fused_1x1_conv_flatten