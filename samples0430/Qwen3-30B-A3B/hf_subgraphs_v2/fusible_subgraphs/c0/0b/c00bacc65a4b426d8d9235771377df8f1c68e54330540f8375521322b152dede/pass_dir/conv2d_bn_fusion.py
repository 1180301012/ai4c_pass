import torch
import triton
import triton.language as tl

def pattern(in_5, in_4, in_0, in_1, in_3, in_2):
    conv2d = torch.conv2d(in_5, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    batch_norm = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return batch_norm

def replacement_args(in_5, in_4, in_0, in_1, in_3, in_2):
    return (in_5, in_4, in_0, in_1, in_3, in_2)

@triton.jit
def conv2d_batch_norm_kernel(
    input_ptr, weight_ptr, running_mean_ptr, running_var_ptr, gamma_ptr, beta_ptr,
    output_ptr,
    N, C_in, C_out, H_in, W_in, H_out, W_out, H_k, W_k, eps,
    stride_h, stride_w, padding_h, padding_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    if pid_n >= N or pid_c >= C_out or pid_h >= H_out or pid_w >= W_out:
        return

    h_out = pid_h
    w_out = pid_w

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for k_h in range(H_k):
        for k_w in range(W_k):
            h_in = h_out * stride_h - padding_h + k_h
            w_in = w_out * stride_w - padding_w + k_w

            if h_in < 0 or h_in >= H_in or w_in < 0 or w_in >= W_in:
                continue

            weight = tl.load(weight_ptr + pid_c * C_in * H_k * W_k + k_h * W_k + k_w)

            for c_in in range(C_in):
                input_val = tl.load(input_ptr + pid_n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in)
                acc += weight * input_val

    scale = gamma_ptr[pid_c] / tl.sqrt(running_var_ptr[pid_c] + eps)
    bias = beta_ptr[pid_c] - gamma_ptr[pid_c] * running_mean_ptr[pid_c] / tl.sqrt(running_var_ptr[pid_c] + eps)

    output = acc * scale + bias

    for h in range(BLOCK_H):
        for w in range(BLOCK_W):
            if h < H_out - pid_h * BLOCK_H and w < W_out - pid_w * BLOCK_W:
                tl.store(output_ptr + pid_n * C_out * H_out * W_out + pid_c * H_out * W_out + (pid_h * BLOCK_H + h) * W_out + (pid_w * BLOCK_W + w),
                         output[h, w])

@torch.fx.wrap
def conv2d_bn(in_5, in_4, in_0, in_1, in_3, in_2):
    N, C_in, H_in, W_in = in_5.shape
    C_out, _, H_k, W_k = in_4.shape

    H_out = H_in - H_k + 2 * 1 + 1
    W_out = W_in - W_k + 2 * 1 + 1

    output = torch.empty((N, C_out, H_out, W_out), device=in_5.device, dtype=in_5.dtype)

    grid = (N, C_out, (H_out + 31) // 32, (W_out + 31) // 32)

    conv2d_batch_norm_kernel[grid](
        in_5, in_4, in_0, in_1, in_3, in_2,
        output,
        N, C_in, C_out, H_in, W_in, H_out, W_out, H_k, W_k, 1e-05,
        1, 1, 1, 1,
        BLOCK_H=32, BLOCK_W=32, BLOCK_C=8
    )

    return output

def replacement_func():
    return conv2d_bn