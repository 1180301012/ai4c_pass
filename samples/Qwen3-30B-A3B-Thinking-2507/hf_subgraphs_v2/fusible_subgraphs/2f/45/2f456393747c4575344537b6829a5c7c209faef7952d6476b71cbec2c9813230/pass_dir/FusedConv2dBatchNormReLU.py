import torch
import triton
import triton.language as tl

def pattern(in_6, in_4, in_0, in_1, in_3, in_2):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    return tmp_7

def replacement_args(in_6, in_4, in_0, in_1, in_3, in_2):
    return (in_6, in_4, in_0, in_1, in_3, in_2)

@triton.jit
def fused_kernel(
    in_ptr, weight_ptr, running_mean_ptr, running_var_ptr, weight_bn_ptr, bias_bn_ptr,
    out_ptr,
    B, C_in, H, W, C_out,
    EPS: tl.constexpr,
    LAMBDA: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr
):
    h = tl.program_id(0)
    w = tl.program_id(1)
    c_out = tl.program_id(2)

    h_start = h * BLOCK_SIZE_H
    w_start = w * BLOCK_SIZE_W
    h_end = min(h_start + BLOCK_SIZE_H, H)
    w_end = min(w_start + BLOCK_SIZE_W, W)

    weight = tl.load(weight_ptr + c_out * (C_in * 9),
                     mask=tl.arange(0, C_in) < C_in,
                     other=0.0)

    out = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for ih in range(BLOCK_SIZE_H):
        for iw in range(BLOCK_SIZE_W):
            for ki in range(3):
                for kj in range(3):
                    ih_in = h_start + ih - 1 + ki
                    iw_in = w_start + iw - 1 + kj
                    if ih_in < 0 or ih_in >= H or iw_in < 0 or iw_in >= W:
                        input_val = 0.0
                    else:
                        input_val = tl.load(in_ptr + (ih_in * W + iw_in) * C_in + tl.arange(0, C_in))
                    w_idx = (ki * 3 + kj) * C_in
                    for c_in in range(C_in):
                        out[ih, iw] += input_val[c_in] * weight[w_idx + c_in]

    running_mean = tl.load(running_mean_ptr + c_out)
    running_var = tl.load(running_var_ptr + c_out)
    weight_bn = tl.load(weight_bn_ptr + c_out)
    bias_bn = tl.load(bias_bn_ptr + c_out)

    for ih in range(BLOCK_SIZE_H):
        for iw in range(BLOCK_SIZE_W):
            val = out[ih, iw]
            val = (val - running_mean) / tl.sqrt(running_var + EPS)
            val = val * weight_bn + bias_bn
            out[ih, iw] = val

    for ih in range(BLOCK_SIZE_H):
        for iw in range(BLOCK_SIZE_W):
            val = out[ih, iw]
            if val < 0:
                val = val * LAMBDA
            out[ih, iw] = val

    out_idx = (h_start * W + w_start) * C_out + c_out
    tl.store(out_ptr + out_idx, out, mask=tl.arange(0, BLOCK_SIZE_H * BLOCK_SIZE_W) < ((h_end - h_start) * (w_end - w_start)))

@torch.fx.wrap
def fused_conv2d_batchnorm_leakyrelu(in_6, in_4, in_0, in_1, in_3, in_2):
    B = in_6.shape[0]
    C_in = in_6.shape[1]
    H = in_6.shape[2]
    W = in_6.shape[3]
    C_out = in_4.shape[0]
    out = torch.empty((B, C_out, H, W), dtype=in_6.dtype, device=in_6.device)

    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    num_blocks_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_blocks_w = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    num_blocks_cout = C_out

    fused_kernel[(num_blocks_h, num_blocks_w, num_blocks_cout)](
        in_6, in_4, in_0, in_1, in_3, in_2,
        out,
        B, C_in, H, W, C_out,
        1e-05,  # EPS
        0.01,   # LAMBDA for LeakyReLU
        BLOCK_SIZE_H,
        BLOCK_SIZE_W
    )
    return out

def replacement_func():
    return fused_conv2d_batchnorm_leakyrelu