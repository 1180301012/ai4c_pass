import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 8}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_warps=8, num_stages=2),
    ],
    key=["h_dim", "w_dim"],
)
@triton.jit
def _depthwise_conv3x3_bias_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    c_dim,
    h_dim,
    w_dim,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    c = pid_nc % c_dim
    base = pid_nc * h_dim * w_dim
    w_base = c * 9

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    acc += tl.load(b_ptr + c).to(tl.float32)

    w00 = tl.load(w_ptr + w_base + 0).to(tl.float32)
    w01 = tl.load(w_ptr + w_base + 1).to(tl.float32)
    w02 = tl.load(w_ptr + w_base + 2).to(tl.float32)
    w10 = tl.load(w_ptr + w_base + 3).to(tl.float32)
    w11 = tl.load(w_ptr + w_base + 4).to(tl.float32)
    w12 = tl.load(w_ptr + w_base + 5).to(tl.float32)
    w20 = tl.load(w_ptr + w_base + 6).to(tl.float32)
    w21 = tl.load(w_ptr + w_base + 7).to(tl.float32)
    w22 = tl.load(w_ptr + w_base + 8).to(tl.float32)

    ih0 = offs_h[:, None] - 1
    ih1 = offs_h[:, None]
    ih2 = offs_h[:, None] + 1
    iw0 = offs_w[None, :] - 1
    iw1 = offs_w[None, :]
    iw2 = offs_w[None, :] + 1

    m00 = (ih0 >= 0) & (ih0 < h_dim) & (iw0 >= 0) & (iw0 < w_dim)
    m01 = (ih0 >= 0) & (ih0 < h_dim) & (iw1 >= 0) & (iw1 < w_dim)
    m02 = (ih0 >= 0) & (ih0 < h_dim) & (iw2 >= 0) & (iw2 < w_dim)
    m10 = (ih1 >= 0) & (ih1 < h_dim) & (iw0 >= 0) & (iw0 < w_dim)
    m11 = (ih1 >= 0) & (ih1 < h_dim) & (iw1 >= 0) & (iw1 < w_dim)
    m12 = (ih1 >= 0) & (ih1 < h_dim) & (iw2 >= 0) & (iw2 < w_dim)
    m20 = (ih2 >= 0) & (ih2 < h_dim) & (iw0 >= 0) & (iw0 < w_dim)
    m21 = (ih2 >= 0) & (ih2 < h_dim) & (iw1 >= 0) & (iw1 < w_dim)
    m22 = (ih2 >= 0) & (ih2 < h_dim) & (iw2 >= 0) & (iw2 < w_dim)

    x00 = tl.load(x_ptr + base + ih0 * w_dim + iw0, mask=m00, other=0.0).to(tl.float32)
    x01 = tl.load(x_ptr + base + ih0 * w_dim + iw1, mask=m01, other=0.0).to(tl.float32)
    x02 = tl.load(x_ptr + base + ih0 * w_dim + iw2, mask=m02, other=0.0).to(tl.float32)
    x10 = tl.load(x_ptr + base + ih1 * w_dim + iw0, mask=m10, other=0.0).to(tl.float32)
    x11 = tl.load(x_ptr + base + ih1 * w_dim + iw1, mask=m11, other=0.0).to(tl.float32)
    x12 = tl.load(x_ptr + base + ih1 * w_dim + iw2, mask=m12, other=0.0).to(tl.float32)
    x20 = tl.load(x_ptr + base + ih2 * w_dim + iw0, mask=m20, other=0.0).to(tl.float32)
    x21 = tl.load(x_ptr + base + ih2 * w_dim + iw1, mask=m21, other=0.0).to(tl.float32)
    x22 = tl.load(x_ptr + base + ih2 * w_dim + iw2, mask=m22, other=0.0).to(tl.float32)

    acc += x00 * w00
    acc += x01 * w01
    acc += x02 * w02
    acc += x10 * w10
    acc += x11 * w11
    acc += x12 * w12
    acc += x20 * w20
    acc += x21 * w21
    acc += x22 * w22

    out = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))
    out_mask = (offs_h[:, None] < h_dim) & (offs_w[None, :] < w_dim)
    tl.store(out_ptr + base + offs_h[:, None] * w_dim + offs_w[None, :], out, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["p_dim", "oc_dim", "c_dim"],
)
@triton.jit
def _pointwise_conv1x1_bias_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    p_dim,
    c_dim,
    oc_dim,
    hw_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    p_mask = offs_m < p_dim
    oc_mask = offs_n < oc_dim

    n_idx = offs_m // hw_dim
    hw_idx = offs_m % hw_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, c_dim, BLOCK_K):
        k_idx = k0 + offs_k
        k_mask = k_idx < c_dim

        x_ptrs = x_ptr + (n_idx[:, None] * c_dim + k_idx[None, :]) * hw_dim + hw_idx[:, None]
        w_ptrs = w_ptr + k_idx[:, None] + offs_n[None, :] * c_dim

        x = tl.load(x_ptrs, mask=p_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & oc_mask[None, :], other=0.0)
        acc += tl.dot(x, w)

    bias = tl.load(b_ptr + offs_n, mask=oc_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]
    out = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))

    out_ptrs = out_ptr + (n_idx[:, None] * oc_dim + offs_n[None, :]) * hw_dim + hw_idx[:, None]
    tl.store(out_ptrs, out, mask=p_mask[:, None] & oc_mask[None, :])


@torch.fx.wrap
def shared_replacement(in_0, in_1, in_2, route):
    out_channels = in_1.shape[0]
    out_shape = (in_2.shape[0], out_channels, in_2.shape[2], in_2.shape[3])
    out = torch.empty(out_shape, device=in_2.device, dtype=in_2.dtype)

    if route == "pw1x1":
        p_dim = in_2.shape[0] * in_2.shape[2] * in_2.shape[3]
        hw_dim = in_2.shape[2] * in_2.shape[3]
        grid = lambda meta: (
            triton.cdiv(p_dim, meta["BLOCK_M"]),
            triton.cdiv(out_channels, meta["BLOCK_N"]),
        )
        _pointwise_conv1x1_bias_gelu_kernel[grid](
            in_2,
            in_1,
            in_0,
            out,
            p_dim,
            in_2.shape[1],
            out_channels,
            hw_dim,
        )
        return out

    grid = lambda meta: (
        in_2.shape[0] * in_2.shape[1],
        triton.cdiv(in_2.shape[2], meta["BLOCK_H"]),
        triton.cdiv(in_2.shape[3], meta["BLOCK_W"]),
    )
    _depthwise_conv3x3_bias_gelu_kernel[grid](
        in_2,
        in_1,
        in_0,
        out,
        in_2.shape[1],
        in_2.shape[2],
        in_2.shape[3],
    )
    return out


def replacement_func():
    return shared_replacement