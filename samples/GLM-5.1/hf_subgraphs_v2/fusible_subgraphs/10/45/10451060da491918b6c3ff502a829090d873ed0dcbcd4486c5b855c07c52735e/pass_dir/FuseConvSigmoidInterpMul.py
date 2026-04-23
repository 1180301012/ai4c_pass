import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Kernel 1: 1x1 conv2d (matmul) + sigmoid
@triton.jit
def conv_sigmoid_kernel(
    weight_ptr, input_ptr, output_ptr,
    M, N, K,
    stride_wm, stride_wk,
    stride_ik, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < M

    off_n = tl.arange(0, BLOCK_N)
    mask_n = off_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        off_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = off_k < K

        w_ptrs = weight_ptr + off_m[:, None] * stride_wm + off_k[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        x_ptrs = input_ptr + off_k[:, None] * stride_ik + off_n[None, :] * stride_in
        x = tl.load(x_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(w, x, allow_tf32=False)

    # Sigmoid: 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-acc))

    tl.store(output_ptr + off_m[:, None] * stride_om + off_n[None, :] * stride_on, sig, mask=mask_m[:, None] & mask_n[None, :])


# Kernel 2: bilinear interpolation + multiply
@triton.jit
def interp_mul_kernel(
    sig_ptr, in2_ptr, out_ptr,
    C, H_out, W_out, W_src, H_src,
    stride_sig_c, stride_sig_h, stride_sig_w,
    stride_in2_n, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    total = C * H_out * W_out
    mask = offs < total

    # Decode flat index to (c, y, x)
    hw = H_out * W_out
    c = offs // hw
    remaining = offs % hw
    y = remaining // W_out
    x = remaining % W_out

    # Bilinear interpolation source coordinates (align_corners=False)
    # src_y = (dst_y + 0.5) * (H_src / H_out) - 0.5
    # src_x = (dst_x + 0.5) * (W_src / W_out) - 0.5
    src_y_float = (y.to(tl.float32) + 0.5) * (H_src.to(tl.float32) / H_out.to(tl.float32)) - 0.5
    src_x_float = (x.to(tl.float32) + 0.5) * (W_src.to(tl.float32) / W_out.to(tl.float32)) - 0.5

    # Floor and clamp
    y0_f = tl.math.floor(src_y_float)
    y0_f = tl.maximum(y0_f, 0.0)
    y0_f = tl.minimum(y0_f, H_src.to(tl.float32) - 1.0)
    y1_f = tl.minimum(y0_f + 1.0, H_src.to(tl.float32) - 1.0)

    x0_f = tl.math.floor(src_x_float)
    x0_f = tl.maximum(x0_f, 0.0)
    x0_f = tl.minimum(x0_f, W_src.to(tl.float32) - 1.0)
    x1_f = tl.minimum(x0_f + 1.0, W_src.to(tl.float32) - 1.0)

    # Interpolation weights
    wy = src_y_float - y0_f
    wy = tl.maximum(wy, 0.0)
    wy = tl.minimum(wy, 1.0)
    wx = src_x_float - x0_f
    wx = tl.maximum(wx, 0.0)
    wx = tl.minimum(wx, 1.0)

    # Convert to int for indexing
    y0 = y0_f.to(tl.int32)
    y1 = y1_f.to(tl.int32)
    x0 = x0_f.to(tl.int32)
    x1 = x1_f.to(tl.int32)

    # Load 4 corner values from sigmoid result [1, C, H_src, W_src]
    v00 = tl.load(sig_ptr + c * stride_sig_c + y0 * stride_sig_h + x0 * stride_sig_w, mask=mask, other=0.0)
    v01 = tl.load(sig_ptr + c * stride_sig_c + y0 * stride_sig_h + x1 * stride_sig_w, mask=mask, other=0.0)
    v10 = tl.load(sig_ptr + c * stride_sig_c + y1 * stride_sig_h + x0 * stride_sig_w, mask=mask, other=0.0)
    v11 = tl.load(sig_ptr + c * stride_sig_c + y1 * stride_sig_h + x1 * stride_sig_w, mask=mask, other=0.0)

    # Bilinear interpolation
    interp_val = (v00 * (1.0 - wy) * (1.0 - wx) +
                  v01 * (1.0 - wy) * wx +
                  v10 * wy * (1.0 - wx) +
                  v11 * wy * wx)

    # Load in_2 value [1, C, H_out, W_out]
    in2_val = tl.load(in2_ptr + 0 * stride_in2_n + c * stride_in2_c + y * stride_in2_h + x * stride_in2_w, mask=mask, other=0.0)

    # Multiply
    out_val = in2_val * interp_val

    # Store
    tl.store(out_ptr + 0 * stride_out_n + c * stride_out_c + y * stride_out_h + x * stride_out_w, out_val, mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_interp_mul(weight, input_tensor, in2):
    # weight shape: [C_out, C_in, 1, 1] = [128, 960, 1, 1]
    # input_tensor shape: [1, C_in, 1, 4] = [1, 960, 1, 4]
    # in2 shape: [1, C_out, 64, 128] = [1, 128, 64, 128]

    C_out = weight.shape[0]  # 128
    C_in = weight.shape[1]   # 960
    H_src = input_tensor.shape[2]  # 1
    W_src = input_tensor.shape[3]  # 4

    H_out = 64
    W_out = 128

    # Reshape for matmul: weight [C_out, C_in], input [C_in, W_src]
    weight_2d = weight.reshape(C_out, C_in)
    input_2d = input_tensor.reshape(C_in, W_src)

    # Allocate sigmoid result: [1, C_out, H_src, W_src]
    sig_result = torch.empty((1, C_out, H_src, W_src), dtype=weight.dtype, device=weight.device)

    # Launch conv_sigmoid_kernel
    # Treat sig_result as 2D [C_out, W_src] for matmul
    sig_2d = sig_result.reshape(C_out, W_src)

    M = C_out
    N = W_src
    K = C_in

    BLOCK_M = 32
    BLOCK_N = 4
    BLOCK_K = 64

    grid_conv = ((M + BLOCK_M - 1) // BLOCK_M,)

    conv_sigmoid_kernel[grid_conv](
        weight_ptr=weight_2d,
        input_ptr=input_2d,
        output_ptr=sig_2d,
        M=M, N=N, K=K,
        stride_wm=weight_2d.stride()[0], stride_wk=weight_2d.stride()[1],
        stride_ik=input_2d.stride()[0], stride_in=input_2d.stride()[1],
        stride_om=sig_2d.stride()[0], stride_on=sig_2d.stride()[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Allocate final output: [1, C_out, H_out, W_out]
    output = torch.empty((1, C_out, H_out, W_out), dtype=in2.dtype, device=in2.device)

    # Launch interp_mul_kernel
    total_elements = C_out * H_out * W_out
    BLOCK_SIZE = 256
    grid_interp = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    interp_mul_kernel[grid_interp](
        sig_ptr=sig_result,
        in2_ptr=in2,
        out_ptr=output,
        C=C_out, H_out=H_out, W_out=W_out, W_src=W_src, H_src=H_src,
        stride_sig_c=sig_result.stride()[1], stride_sig_h=sig_result.stride()[2], stride_sig_w=sig_result.stride()[3],
        stride_in2_n=in2.stride()[0], stride_in2_c=in2.stride()[1], stride_in2_h=in2.stride()[2], stride_in2_w=in2.stride()[3],
        stride_out_n=output.stride()[0], stride_out_c=output.stride()[1], stride_out_h=output.stride()[2], stride_out_w=output.stride()[3],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (output,)


def replacement_func():
    return fused_conv_sigmoid_interp_mul