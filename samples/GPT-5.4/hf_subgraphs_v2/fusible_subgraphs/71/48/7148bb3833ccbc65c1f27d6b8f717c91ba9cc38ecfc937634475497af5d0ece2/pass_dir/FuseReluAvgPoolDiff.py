import torch
import triton
import triton.language as tl


def pattern(in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    return tmp_2, tmp_4


def replacement_args(in_2):
    return (in_2,)


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
def fused_relu_avgpool_diff_kernel(
    x_ptr,
    relu_ptr,
    diff_ptr,
    N,
    C,
    H,
    W,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    relu_stride_n,
    relu_stride_c,
    relu_stride_h,
    relu_stride_w,
    diff_stride_n,
    diff_stride_c,
    diff_stride_h,
    diff_stride_w,
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
    base_relu = n * relu_stride_n + c * relu_stride_c
    base_diff = n * diff_stride_n + c * diff_stride_c

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

    center_idx = base_x + offs_h * x_stride_h + offs_w * x_stride_w
    center = tl.load(x_ptr + center_idx, mask=out_mask, other=0.0)
    center = tl.maximum(center.to(tl.float32), 0.0)
    avg = sum_val / cnt_val
    diff = avg - center

    relu_idx = base_relu + offs_h * relu_stride_h + offs_w * relu_stride_w
    diff_idx = base_diff + offs_h * diff_stride_h + offs_w * diff_stride_w
    tl.store(relu_ptr + relu_idx, center, mask=out_mask)
    tl.store(diff_ptr + diff_idx, diff, mask=out_mask)


@torch.fx.wrap
def fused_relu_avgpool_diff(in_2):
    relu_out = torch.empty_like(in_2)
    diff_out = torch.empty_like(in_2)

    N = in_2.shape[0]
    C = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]

    x_stride_n, x_stride_c, x_stride_h, x_stride_w = in_2.stride()
    relu_stride_n, relu_stride_c, relu_stride_h, relu_stride_w = relu_out.stride()
    diff_stride_n, diff_stride_c, diff_stride_h, diff_stride_w = diff_out.stride()

    grid = lambda meta: (
        triton.cdiv(W, meta["BLOCK_W"]),
        triton.cdiv(H, meta["BLOCK_H"]),
        N * C,
    )

    fused_relu_avgpool_diff_kernel[grid](
        in_2,
        relu_out,
        diff_out,
        N,
        C,
        H,
        W,
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w,
        relu_stride_n,
        relu_stride_c,
        relu_stride_h,
        relu_stride_w,
        diff_stride_n,
        diff_stride_c,
        diff_stride_h,
        diff_stride_w,
    )
    return relu_out, diff_out


def replacement_func():
    return fused_relu_avgpool_diff