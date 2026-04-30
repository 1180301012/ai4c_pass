import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_nchw_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    H,
    W,
    HW,
    xs0,
    xs1,
    xs2,
    xs3,
    ws0,
    ws1,
    bs0,
    ys0,
    ys1,
    ys2,
    ys3,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    batch_idx = offs_m // HW
    rem = offs_m % HW
    h_idx = rem // W
    w_idx = rem % W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        x_ptrs = (
            x_ptr
            + batch_idx[:, None] * xs0
            + k_idx[None, :] * xs1
            + h_idx[:, None] * xs2
            + w_idx[:, None] * xs3
        )
        w_ptrs = w_ptr + offs_n[None, :] * ws0 + k_idx[:, None] * ws1

        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)

    bias = tl.load(b_ptr + offs_n * bs0, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + batch_idx[:, None] * ys0
        + offs_n[None, :] * ys1
        + h_idx[:, None] * ys2
        + w_idx[:, None] * ys3
    )
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
    ],
    key=['TOTAL'],
)
@triton.jit
def upsample_bilinear_nchw_kernel(
    x_ptr,
    y_ptr,
    B,
    C,
    H_IN,
    W_IN,
    H_OUT,
    W_OUT,
    xs0,
    xs1,
    xs2,
    xs3,
    ys0,
    ys1,
    ys2,
    ys3,
    TOTAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    ow = offs % W_OUT
    t0 = offs // W_OUT
    oh = t0 % H_OUT
    t1 = t0 // H_OUT
    c = t1 % C
    b = t1 // C

    scale_h = (H_IN * 1.0) / H_OUT
    scale_w = (W_IN * 1.0) / W_OUT

    in_y = (oh.to(tl.float32) + 0.5) * scale_h - 0.5
    in_x = (ow.to(tl.float32) + 0.5) * scale_w - 0.5

    y0 = tl.floor(in_y).to(tl.int32)
    x0 = tl.floor(in_x).to(tl.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    ly = in_y - y0.to(tl.float32)
    lx = in_x - x0.to(tl.float32)
    hy = 1.0 - ly
    hx = 1.0 - lx

    y0c = tl.minimum(tl.maximum(y0, 0), H_IN - 1)
    x0c = tl.minimum(tl.maximum(x0, 0), W_IN - 1)
    y1c = tl.minimum(tl.maximum(y1, 0), H_IN - 1)
    x1c = tl.minimum(tl.maximum(x1, 0), W_IN - 1)

    base = b * xs0 + c * xs1
    p00 = x_ptr + base + y0c * xs2 + x0c * xs3
    p01 = x_ptr + base + y0c * xs2 + x1c * xs3
    p10 = x_ptr + base + y1c * xs2 + x0c * xs3
    p11 = x_ptr + base + y1c * xs2 + x1c * xs3

    v00 = tl.load(p00, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(p01, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(p10, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(p11, mask=mask, other=0.0).to(tl.float32)

    out = v00 * hy * hx + v01 * hy * lx + v10 * ly * hx + v11 * ly * lx

    y_ptrs = y_ptr + b * ys0 + c * ys1 + oh * ys2 + ow * ys3
    tl.store(y_ptrs, out, mask=mask)


@torch.fx.wrap
def decode_head_only(in_10, in_8, in_7):
    bsz, cin, h, w = in_10.shape
    cout = in_8.shape[0]

    conv_out = torch.empty((bsz, cout, h, w), device=in_10.device, dtype=in_10.dtype)
    m = bsz * h * w
    n = cout
    k = cin

    conv_grid = lambda META: (triton.cdiv(m, META['BLOCK_M']) * triton.cdiv(n, META['BLOCK_N']),)
    conv1x1_nchw_kernel[conv_grid](
        in_10,
        in_8,
        in_7,
        conv_out,
        m,
        n,
        k,
        h,
        w,
        h * w,
        in_10.stride(0),
        in_10.stride(1),
        in_10.stride(2),
        in_10.stride(3),
        in_8.stride(0),
        in_8.stride(1),
        in_7.stride(0),
        conv_out.stride(0),
        conv_out.stride(1),
        conv_out.stride(2),
        conv_out.stride(3),
    )

    out_h = 512
    out_w = 512
    out = torch.empty((bsz, cout, out_h, out_w), device=in_10.device, dtype=in_10.dtype)
    total = bsz * cout * out_h * out_w

    upsample_grid = lambda META: (triton.cdiv(total, META['BLOCK']),)
    upsample_bilinear_nchw_kernel[upsample_grid](
        conv_out,
        out,
        bsz,
        cout,
        h,
        w,
        out_h,
        out_w,
        conv_out.stride(0),
        conv_out.stride(1),
        conv_out.stride(2),
        conv_out.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        total,
    )

    return (out,)