import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def bn_only_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    H,
    W,
    x_sc,
    x_sh,
    x_sw,
    out_sc,
    out_sh,
    out_sw,
    N,
    EPS,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    hw = H * W
    c = offs // hw
    rem = offs - c * hw
    h = rem // W
    w = rem - h * W

    x_ptrs = x_ptr + c * x_sc + h * x_sh + w * x_sw
    out_ptrs = out_ptr + c * out_sc + h * out_sh + w * out_sw

    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0).to(tl.float32)

    y = ((x - mean) / tl.sqrt(var + EPS)) * weight + bias
    tl.store(out_ptrs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def relu_only_kernel(
    x_ptr,
    out_ptr,
    H,
    W,
    x_sc,
    x_sh,
    x_sw,
    out_sc,
    out_sh,
    out_sw,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    hw = H * W
    c = offs // hw
    rem = offs - c * hw
    h = rem // W
    w = rem - h * W

    x_ptrs = x_ptr + c * x_sc + h * x_sh + w * x_sw
    out_ptrs = out_ptr + c * out_sc + h * out_sh + w * out_sw
    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptrs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def bn_relu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    H,
    W,
    x_sc,
    x_sh,
    x_sw,
    out_sc,
    out_sh,
    out_sw,
    N,
    EPS,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    hw = H * W
    c = offs // hw
    rem = offs - c * hw
    h = rem // W
    w = rem - h * W

    x_ptrs = x_ptr + c * x_sc + h * x_sh + w * x_sw
    out_ptrs = out_ptr + c * out_sc + h * out_sh + w * out_sw

    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0).to(tl.float32)

    y = ((x - mean) / tl.sqrt(var + EPS)) * weight + bias
    y = tl.maximum(y, 0.0)
    tl.store(out_ptrs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["TOTAL_ELEMS"],
)
@triton.jit
def cat5_relu_last_kernel(
    in5_ptr,
    in6_ptr,
    in7_ptr,
    in8_ptr,
    in9_ptr,
    out_ptr,
    C0,
    C1,
    C2,
    C3,
    C4,
    H,
    W,
    in5_sc,
    in5_sh,
    in5_sw,
    in6_sc,
    in6_sh,
    in6_sw,
    in7_sc,
    in7_sh,
    in7_sw,
    in8_sc,
    in8_sh,
    in8_sw,
    in9_sc,
    in9_sh,
    in9_sw,
    out_sc,
    out_sh,
    out_sw,
    TOTAL_ELEMS,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS

    hw = H * W
    out_c = offs // hw
    rem = offs - out_c * hw
    h = rem // W
    w = rem - h * W

    out_ptrs = out_ptr + out_c * out_sc + h * out_sh + w * out_sw

    r0 = mask & (out_c < C0)
    src0 = in5_ptr + out_c * in5_sc + h * in5_sh + w * in5_sw
    v0 = tl.load(src0, mask=r0, other=0)
    tl.store(out_ptrs, v0, mask=r0)

    base1 = C0
    lim1 = C0 + C1
    r1 = mask & (out_c >= base1) & (out_c < lim1)
    c1 = out_c - base1
    src1 = in7_ptr + c1 * in7_sc + h * in7_sh + w * in7_sw
    v1 = tl.load(src1, mask=r1, other=0)
    tl.store(out_ptrs, v1, mask=r1)

    base2 = lim1
    lim2 = lim1 + C2
    r2 = mask & (out_c >= base2) & (out_c < lim2)
    c2 = out_c - base2
    src2 = in8_ptr + c2 * in8_sc + h * in8_sh + w * in8_sw
    v2 = tl.load(src2, mask=r2, other=0)
    tl.store(out_ptrs, v2, mask=r2)

    base3 = lim2
    lim3 = lim2 + C3
    r3 = mask & (out_c >= base3) & (out_c < lim3)
    c3 = out_c - base3
    src3 = in6_ptr + c3 * in6_sc + h * in6_sh + w * in6_sw
    v3 = tl.load(src3, mask=r3, other=0)
    tl.store(out_ptrs, v3, mask=r3)

    base4 = lim3
    r4 = mask & (out_c >= base4) & (out_c < (base4 + C4))
    c4 = out_c - base4
    src4 = in9_ptr + c4 * in9_sc + h * in9_sh + w * in9_sw
    v4 = tl.load(src4, mask=r4, other=0).to(tl.float32)
    tl.store(out_ptrs, tl.maximum(v4, 0.0), mask=r4)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["TOTAL_ELEMS"],
)
@triton.jit
def cat5_kernel(
    in5_ptr,
    in6_ptr,
    in7_ptr,
    in8_ptr,
    in9_ptr,
    out_ptr,
    C0,
    C1,
    C2,
    C3,
    C4,
    H,
    W,
    in5_sc,
    in5_sh,
    in5_sw,
    in6_sc,
    in6_sh,
    in6_sw,
    in7_sc,
    in7_sh,
    in7_sw,
    in8_sc,
    in8_sh,
    in8_sw,
    in9_sc,
    in9_sh,
    in9_sw,
    out_sc,
    out_sh,
    out_sw,
    TOTAL_ELEMS,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS

    hw = H * W
    out_c = offs // hw
    rem = offs - out_c * hw
    h = rem // W
    w = rem - h * W

    out_ptrs = out_ptr + out_c * out_sc + h * out_sh + w * out_sw

    r0 = mask & (out_c < C0)
    src0 = in5_ptr + out_c * in5_sc + h * in5_sh + w * in5_sw
    v0 = tl.load(src0, mask=r0, other=0)
    tl.store(out_ptrs, v0, mask=r0)

    base1 = C0
    lim1 = C0 + C1
    r1 = mask & (out_c >= base1) & (out_c < lim1)
    c1 = out_c - base1
    src1 = in7_ptr + c1 * in7_sc + h * in7_sh + w * in7_sw
    v1 = tl.load(src1, mask=r1, other=0)
    tl.store(out_ptrs, v1, mask=r1)

    base2 = lim1
    lim2 = lim1 + C2
    r2 = mask & (out_c >= base2) & (out_c < lim2)
    c2 = out_c - base2
    src2 = in8_ptr + c2 * in8_sc + h * in8_sh + w * in8_sw
    v2 = tl.load(src2, mask=r2, other=0)
    tl.store(out_ptrs, v2, mask=r2)

    base3 = lim2
    lim3 = lim2 + C3
    r3 = mask & (out_c >= base3) & (out_c < lim3)
    c3 = out_c - base3
    src3 = in6_ptr + c3 * in6_sc + h * in6_sh + w * in6_sw
    v3 = tl.load(src3, mask=r3, other=0)
    tl.store(out_ptrs, v3, mask=r3)

    base4 = lim3
    r4 = mask & (out_c >= base4) & (out_c < (base4 + C4))
    c4 = out_c - base4
    src4 = in9_ptr + c4 * in9_sc + h * in9_sh + w * in9_sw
    v4 = tl.load(src4, mask=r4, other=0)
    tl.store(out_ptrs, v4, mask=r4)


@torch.fx.wrap
def shared_dmnet_dispatch(*args):
    route = args[-1]

    if route == "view":
        x = args[0]
        return x

    if route == "bn":
        x, mean, var, bias, weight = args[:-1]
        h = x.shape[2]
        w = x.shape[3]
        c = x.shape[1]
        out = torch.empty((1, c, h, w), device=x.device, dtype=x.dtype)
        n = c * h * w
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        bn_only_kernel[grid](
            x,
            mean,
            var,
            bias,
            weight,
            out,
            h,
            w,
            x.stride(1),
            x.stride(2),
            x.stride(3),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            n,
            1e-05,
        )
        return out

    if route == "relu":
        x = args[0]
        h = x.shape[2]
        w = x.shape[3]
        c = x.shape[1]
        out = torch.empty((1, c, h, w), device=x.device, dtype=x.dtype)
        n = c * h * w
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        relu_only_kernel[grid](
            x,
            out,
            h,
            w,
            x.stride(1),
            x.stride(2),
            x.stride(3),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            n,
        )
        return out

    if route == "bn_relu":
        x, mean, var, bias, weight = args[:-1]
        h = x.shape[2]
        w = x.shape[3]
        c = x.shape[1]
        out = torch.empty((1, c, h, w), device=x.device, dtype=x.dtype)
        n = c * h * w
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        bn_relu_kernel[grid](
            x,
            mean,
            var,
            bias,
            weight,
            out,
            h,
            w,
            x.stride(1),
            x.stride(2),
            x.stride(3),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            n,
            1e-05,
        )
        return out

    if route == "cat5_relu_last":
        in_5, in_6, in_7, in_8, tmp_6 = args[:-1]
        h = in_6.shape[2]
        w = in_6.shape[3]
        c0 = in_5.shape[1]
        c1 = in_7.shape[1]
        c2 = in_8.shape[1]
        c3 = in_6.shape[1]
        c4 = tmp_6.shape[1]
        out_c = c0 + c1 + c2 + c3 + c4
        out = torch.empty((1, out_c, h, w), device=in_5.device, dtype=in_5.dtype)
        total_elems = out_c * h * w
        grid = lambda meta: (triton.cdiv(total_elems, meta["BLOCK"]),)
        cat5_relu_last_kernel[grid](
            in_5,
            in_6,
            in_7,
            in_8,
            tmp_6,
            out,
            c0,
            c1,
            c2,
            c3,
            c4,
            h,
            w,
            in_5.stride(1),
            in_5.stride(2),
            in_5.stride(3),
            in_6.stride(1),
            in_6.stride(2),
            in_6.stride(3),
            in_7.stride(1),
            in_7.stride(2),
            in_7.stride(3),
            in_8.stride(1),
            in_8.stride(2),
            in_8.stride(3),
            tmp_6.stride(1),
            tmp_6.stride(2),
            tmp_6.stride(3),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            total_elems,
        )
        return out

    if route == "cat5":
        in_5, in_6, in_7, in_8, tmp_7 = args[:-1]
        h = in_6.shape[2]
        w = in_6.shape[3]
        c0 = in_5.shape[1]
        c1 = in_7.shape[1]
        c2 = in_8.shape[1]
        c3 = in_6.shape[1]
        c4 = tmp_7.shape[1]
        out_c = c0 + c1 + c2 + c3 + c4
        out = torch.empty((1, out_c, h, w), device=in_5.device, dtype=in_5.dtype)
        total_elems = out_c * h * w
        grid = lambda meta: (triton.cdiv(total_elems, meta["BLOCK"]),)
        cat5_kernel[grid](
            in_5,
            in_6,
            in_7,
            in_8,
            tmp_7,
            out,
            c0,
            c1,
            c2,
            c3,
            c4,
            h,
            w,
            in_5.stride(1),
            in_5.stride(2),
            in_5.stride(3),
            in_6.stride(1),
            in_6.stride(2),
            in_6.stride(3),
            in_7.stride(1),
            in_7.stride(2),
            in_7.stride(3),
            in_8.stride(1),
            in_8.stride(2),
            in_8.stride(3),
            tmp_7.stride(1),
            tmp_7.stride(2),
            tmp_7.stride(3),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            total_elems,
        )
        return out

    raise RuntimeError(f"Unknown route: {route}")