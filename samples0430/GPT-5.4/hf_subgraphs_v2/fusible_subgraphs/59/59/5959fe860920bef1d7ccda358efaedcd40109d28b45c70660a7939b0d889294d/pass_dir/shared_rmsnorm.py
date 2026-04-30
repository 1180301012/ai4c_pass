import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_2048_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    eps,
    OUT_FP32: tl.constexpr,
    USE_WEIGHT: tl.constexpr,
    NUM_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    base = row * NUM_COLS
    offs = tl.arange(0, HALF_COLS)

    x0 = tl.load(x_ptr + base + offs).to(tl.float32)
    x1 = tl.load(x_ptr + base + HALF_COLS + offs).to(tl.float32)

    ss = tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0)
    inv_rms = tl.rsqrt(ss / NUM_COLS + eps)

    if OUT_FP32:
        y0 = x0 * inv_rms
        y1 = x1 * inv_rms
        if USE_WEIGHT:
            w0 = tl.load(w_ptr + offs).to(tl.float32)
            w1 = tl.load(w_ptr + HALF_COLS + offs).to(tl.float32)
            y0 = y0 * w0
            y1 = y1 * w1
        tl.store(out_ptr + base + offs, y0)
        tl.store(out_ptr + base + HALF_COLS + offs, y1)
    else:
        n0 = (x0 * inv_rms).to(tl.bfloat16)
        n1 = (x1 * inv_rms).to(tl.bfloat16)
        if USE_WEIGHT:
            w0 = tl.load(w_ptr + offs)
            w1 = tl.load(w_ptr + HALF_COLS + offs)
            y0 = (n0 * w0).to(tl.bfloat16)
            y1 = (n1 * w1).to(tl.bfloat16)
        else:
            y0 = n0
            y1 = n1
        tl.store(out_ptr + base + offs, y0)
        tl.store(out_ptr + base + HALF_COLS + offs, y1)


@triton.jit
def _cat_trig_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    n_rows,
    IN_COLS: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    x_base = row * IN_COLS
    y_base = row * (2 * IN_COLS)
    offs = tl.arange(0, IN_COLS)

    x = tl.load(x_ptr + x_base + offs).to(tl.float32)
    c = tl.cos(x)
    s = tl.sin(x)

    if not OUT_FP32:
        c = c.to(tl.bfloat16)
        s = s.to(tl.bfloat16)

    tl.store(cos_ptr + y_base + offs, c)
    tl.store(cos_ptr + y_base + IN_COLS + offs, c)
    tl.store(sin_ptr + y_base + offs, s)
    tl.store(sin_ptr + y_base + IN_COLS + offs, s)


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]

    if route == "rmsnorm_bf16_eps1e6":
        w, x, _ = args
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        out = torch.empty_like(x)
        _rmsnorm_2048_kernel[(n_rows,)](
            x,
            w,
            out,
            n_rows,
            1e-6,
            OUT_FP32=False,
            USE_WEIGHT=True,
            NUM_COLS=2048,
            HALF_COLS=1024,
            num_warps=8,
            num_stages=2,
        )
        return out

    if route == "rmsnorm_fp32_eps1e5":
        w, x, _ = args
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        _rmsnorm_2048_kernel[(n_rows,)](
            x,
            w,
            out,
            n_rows,
            1e-5,
            OUT_FP32=True,
            USE_WEIGHT=True,
            NUM_COLS=2048,
            HALF_COLS=1024,
            num_warps=8,
            num_stages=2,
        )
        return out

    if route == "rmsnorm_bf16_eps1e6_noweight":
        w, x, _ = args
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        out = torch.empty_like(x)
        _rmsnorm_2048_kernel[(n_rows,)](
            x,
            w,
            out,
            n_rows,
            1e-6,
            OUT_FP32=False,
            USE_WEIGHT=False,
            NUM_COLS=2048,
            HALF_COLS=1024,
            num_warps=8,
            num_stages=2,
        )
        return out

    if route == "rmsnorm_fp32_eps1e5_noweight":
        w, x, _ = args
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        _rmsnorm_2048_kernel[(n_rows,)](
            x,
            w,
            out,
            n_rows,
            1e-5,
            OUT_FP32=True,
            USE_WEIGHT=False,
            NUM_COLS=2048,
            HALF_COLS=1024,
            num_warps=8,
            num_stages=2,
        )
        return out

    if route == "cat_trig_bf16":
        x, _ = args
        in_cols = x.shape[-1]
        n_rows = x.numel() // in_cols
        out_shape = (*x.shape[:-1], in_cols * 2)
        out_cos = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        out_sin = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        _cat_trig_kernel[(n_rows,)](
            x,
            out_cos,
            out_sin,
            n_rows,
            IN_COLS=in_cols,
            OUT_FP32=False,
            num_warps=1,
            num_stages=1,
        )
        return out_cos, out_sin

    if route == "cat_trig_fp32":
        x, _ = args
        in_cols = x.shape[-1]
        n_rows = x.numel() // in_cols
        out_shape = (*x.shape[:-1], in_cols * 2)
        out_cos = torch.empty(out_shape, device=x.device, dtype=torch.float32)
        out_sin = torch.empty(out_shape, device=x.device, dtype=torch.float32)
        _cat_trig_kernel[(n_rows,)](
            x,
            out_cos,
            out_sin,
            n_rows,
            IN_COLS=in_cols,
            OUT_FP32=True,
            num_warps=1,
            num_stages=1,
        )
        return out_cos, out_sin

    raise RuntimeError(f"Unknown route: {route}")