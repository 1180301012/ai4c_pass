import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_9):
    conv2d = torch.conv2d(input=in_9, weight=in_4, groups=512)
    tmp_5 = conv2d.view(1, 512, 64, 64)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_9):
    return (in_0, in_1, in_2, in_3, in_4, in_9)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_warps=4),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 4}, num_warps=4),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 8}, num_warps=4),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 4}, num_warps=2),
    ],
    key=["H_OUT", "W_OUT"],
)
@triton.jit

def _depthwise_conv7x7_bn_relu_kernel(
    x_ptr,
    w_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wc,
    stride_wh,
    stride_ww,
    stride_oc,
    stride_oh,
    stride_ow,
    H_OUT,
    W_OUT,
    eps,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)

    num_tiles_w = tl.cdiv(W_OUT, BLOCK_W)
    tile_h = pid_hw // num_tiles_w
    tile_w = pid_hw % num_tiles_w

    oh = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = tile_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (oh[:, None] < H_OUT) & (ow[None, :] < W_OUT)

    x_c_ptr = x_ptr + pid_c * stride_xc
    w_c_ptr = w_ptr + pid_c * stride_wc
    out_c_ptr = out_ptr + pid_c * stride_oc

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for kh in range(7):
        ih = oh[:, None] + kh
        for kw in range(7):
            iw = ow[None, :] + kw
            x = tl.load(x_c_ptr + ih * stride_xh + iw * stride_xw, mask=mask, other=0.0)
            w = tl.load(w_c_ptr + kh * stride_wh + kw * stride_ww)
            acc += x.to(tl.float32) * w.to(tl.float32)

    mean = tl.load(mean_ptr + pid_c).to(tl.float32)
    var = tl.load(var_ptr + pid_c).to(tl.float32)
    gamma = tl.load(gamma_ptr + pid_c).to(tl.float32)
    beta = tl.load(beta_ptr + pid_c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (acc - mean) * inv_std
    y = y * gamma + beta
    y = tl.maximum(y, 0.0)

    tl.store(out_c_ptr + oh[:, None] * stride_oh + ow[None, :] * stride_ow, y, mask=mask)


@torch.fx.wrap
def fused_depthwise_conv7x7_bn_relu(in_0, in_1, in_2, in_3, in_4, in_9):
    n = in_9.shape[0]
    c = in_9.shape[1]
    h_out = in_9.shape[2] - in_4.shape[2] + 1
    w_out = in_9.shape[3] - in_4.shape[3] + 1

    out = torch.empty((n, c, h_out, w_out), device=in_9.device, dtype=in_9.dtype)

    grid = lambda meta: (
        triton.cdiv(h_out, meta["BLOCK_H"]) * triton.cdiv(w_out, meta["BLOCK_W"]),
        c,
    )

    _depthwise_conv7x7_bn_relu_kernel[grid](
        in_9,
        in_4,
        in_0,
        in_1,
        in_3,
        in_2,
        out,
        in_9.stride(1),
        in_9.stride(2),
        in_9.stride(3),
        in_4.stride(0),
        in_4.stride(2),
        in_4.stride(3),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        h_out,
        w_out,
        1e-5,
    )
    return out


def replacement_func():
    return fused_depthwise_conv7x7_bn_relu