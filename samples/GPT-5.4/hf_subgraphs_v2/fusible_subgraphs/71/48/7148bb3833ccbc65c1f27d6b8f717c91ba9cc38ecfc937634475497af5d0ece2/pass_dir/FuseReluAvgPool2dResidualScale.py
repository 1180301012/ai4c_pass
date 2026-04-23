import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = tmp_2 + tmp_7
    tmp_9 = in_1.unsqueeze(-1)
    tmp_10 = tmp_9.unsqueeze(-1)
    return tmp_8, tmp_10


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_warps=8, num_stages=2),
    ],
    key=["H", "W"],
)
@triton.jit
def fused_relu_avgpool_residual_scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    scale_stride_c,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    c = pid_nc % C
    n = pid_nc // C

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    out_mask = (offs_h < H) & (offs_w < W)

    base_x = n * x_stride_n + c * x_stride_c
    base_out = n * out_stride_n + c * out_stride_c

    sum_val = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    cnt_val = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    for dh in (-1, 0, 1):
        h = offs_h + dh
        h_valid = (h >= 0) & (h < H)
        h_safe = tl.minimum(tl.maximum(h, 0), H - 1)
        for dw in (-1, 0, 1):
            w = offs_w + dw
            w_valid = (w >= 0) & (w < W)
            w_safe = tl.minimum(tl.maximum(w, 0), W - 1)
            valid = h_valid & w_valid
            x_idx = base_x + h_safe * x_stride_h + w_safe * x_stride_w
            x = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            x = tl.maximum(x.to(tl.float32), 0.0)
            sum_val += x
            cnt_val += tl.where(valid, 1.0, 0.0)

    cnt_val = tl.where(out_mask, cnt_val, 1.0)

    center_idx = base_x + offs_h * x_stride_h + offs_w * x_stride_w
    center = tl.load(x_ptr + center_idx, mask=out_mask, other=0.0)
    center = tl.maximum(center.to(tl.float32), 0.0)

    scale = tl.load(scale_ptr + c * scale_stride_c).to(tl.float32)
    avg = sum_val / cnt_val
    out = center + scale * (avg - center)

    out_idx = base_out + offs_h * out_stride_h + offs_w * out_stride_w
    tl.store(out_ptr + out_idx, out, mask=out_mask)


@triton.jit
def unsqueeze_copy_kernel(
    src_ptr,
    out_ptr,
    C,
    src_stride_c,
    out_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < C
    vals = tl.load(src_ptr + offs * src_stride_c, mask=mask, other=0.0)
    tl.store(out_ptr + offs * out_stride_c, vals, mask=mask)


@torch.fx.wrap
def fused_full_graph(in_0, in_1, in_2):
    N = in_2.shape[0]
    C = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]

    out0 = torch.empty((N, C, H, W), device=in_2.device, dtype=in_2.dtype)
    out1 = torch.empty((C, 1, 1), device=in_1.device, dtype=in_1.dtype)

    x_stride_n, x_stride_c, x_stride_h, x_stride_w = in_2.stride()
    scale_stride_c = in_0.stride()[0]
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out0.stride()

    grid = lambda meta: (
        triton.cdiv(W, meta["BLOCK_W"]),
        triton.cdiv(H, meta["BLOCK_H"]),
        N * C,
    )

    fused_relu_avgpool_residual_scale_kernel[grid](
        in_2,
        in_0,
        out0,
        N,
        C,
        H,
        W,
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w,
        scale_stride_c,
        out_stride_n,
        out_stride_c,
        out_stride_h,
        out_stride_w,
    )

    src_stride_c = in_1.stride()[0]
    out1_stride_c = out1.stride()[0]
    BLOCK_SIZE = 64
    unsqueeze_copy_kernel[(triton.cdiv(C, BLOCK_SIZE),)](
        in_1,
        out1,
        C,
        src_stride_c,
        out1_stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out0, out1


def replacement_func():
    return fused_full_graph