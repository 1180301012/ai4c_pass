import torch
import triton
import triton.language as tl


def pattern(input_tensor, weight, bias, layer_scale, residual):
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    dropped = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    scaled = dropped * layer_scale
    result = residual + scaled
    return result


def replacement_args(input_tensor, weight, bias, layer_scale, residual):
    return (input_tensor, weight, bias, layer_scale, residual)


@triton.jit
def fused_conv1x1_scale_add_kernel(
    input_ptr, weight_ptr, bias_ptr, scale_ptr, residual_ptr, output_ptr,
    B, C_in: tl.constexpr, C_out: tl.constexpr, HW,
    input_batch_stride, input_c_stride,
    res_batch_stride, res_c_stride,
    out_batch_stride, out_c_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program computes a [BLOCK_M, BLOCK_N] tile of the output
    # Grid: (B * ceil(HW/BLOCK_M), ceil(C_out/BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_m_per_batch = tl.cdiv(HW, BLOCK_M)
    batch_id = pid_m // num_m_per_batch
    m_tile_id = pid_m % num_m_per_batch

    m_start = m_tile_id * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointer for input for this batch
    input_batch_base = input_ptr + batch_id * input_batch_stride

    # Loop over K dimension (input channels)
    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K] from input
        # input[batch, k, m] at: input_batch_base + k*HW + m
        a_ptrs = input_batch_base + k_offs[None, :] * input_c_stride + m_offs[:, None]
        a_mask = (m_offs[:, None] < HW) & (k_offs[None, :] < C_in)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B tile [BLOCK_K, BLOCK_N] from weight
        # weight[n, k] at: n*C_in + k
        b_ptrs = weight_ptr + n_offs[None, :] * C_in + k_offs[:, None]
        b_mask = (k_offs[:, None] < C_in) & (n_offs[None, :] < C_out)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Add bias: acc += bias[n]
    bias_mask = n_offs < C_out
    bias_vals = tl.load(bias_ptr + n_offs, mask=bias_mask, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # Apply layer_scale: acc *= scale[n]
    scale_vals = tl.load(scale_ptr + n_offs, mask=bias_mask, other=0.0).to(tl.float32)
    acc *= scale_vals[None, :]

    # Add residual: acc += residual[batch, n, m]
    res_ptrs = residual_ptr + batch_id * res_batch_stride + n_offs[None, :] * res_c_stride + m_offs[:, None]
    res_mask = (m_offs[:, None] < HW) & (n_offs[None, :] < C_out)
    res_vals = tl.load(res_ptrs, mask=res_mask, other=0.0).to(tl.float32)
    acc += res_vals

    # Store output: output[batch, n, m]
    out_ptrs = output_ptr + batch_id * out_batch_stride + n_offs[None, :] * out_c_stride + m_offs[:, None]
    out_mask = (m_offs[:, None] < HW) & (n_offs[None, :] < C_out)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_scale_add(input_tensor, weight, bias, layer_scale, residual):
    B = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    C_out = weight.shape[0]
    HW = H * W

    out = torch.empty_like(residual)

    # Strides for NCHW layout
    input_batch_stride = C_in * HW
    input_c_stride = HW
    res_batch_stride = C_out * HW
    res_c_stride = HW
    out_batch_stride = C_out * HW
    out_c_stride = HW

    # Choose tile sizes based on problem size
    if B * HW > 50000:
        BLOCK_M = 128
    else:
        BLOCK_M = 64
    BLOCK_N = 64  # C_out = 64, so one tile covers all output channels
    BLOCK_K = 64

    num_m_tiles = (HW + BLOCK_M - 1) // BLOCK_M
    num_n_tiles = (C_out + BLOCK_N - 1) // BLOCK_N

    grid = (B * num_m_tiles, num_n_tiles)

    fused_conv1x1_scale_add_kernel[grid](
        input_tensor, weight, bias, layer_scale, residual, out,
        B, C_in, C_out, HW,
        input_batch_stride, input_c_stride,
        res_batch_stride, res_c_stride,
        out_batch_stride, out_c_stride,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out


def replacement_func():
    return fused_conv1x1_scale_add