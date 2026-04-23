import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_mul_kernel_3d(
    x_ptr,
    w_ptr,
    out_ptr,
    D0,
    D1,
    sx0,
    sx1,
    sx2,
    so0,
    so1,
    so2,
    eps,
    OUT_FP32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    i0 = pid // D1
    i1 = pid % D1
    x_base = i0 * sx0 + i1 * sx1
    o_base = i0 * so0 + i1 * so1

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start in tl.static_range(0, 2048, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        x = tl.load(x_ptr + x_base + offs * sx2).to(tl.float32)
        acc += x * x

    mean_sq = tl.sum(acc, axis=0) / 2048.0
    rstd = tl.rsqrt(mean_sq + eps)

    for start in tl.static_range(0, 2048, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        x = tl.load(x_ptr + x_base + offs * sx2).to(tl.float32)
        w = tl.load(w_ptr + offs)
        if OUT_FP32:
            out = (x * rstd) * w.to(tl.float32)
        else:
            tmp16 = (x * rstd).to(tl.bfloat16)
            out = (tmp16 * w).to(tl.bfloat16)
        tl.store(out_ptr + o_base + offs * so2, out)


@torch.fx.wrap
def fused_dispatch(*args):
    route = args[-1]

    if route == "smollm_bf16":
        w, x = args[0], args[1]
        D0, D1, N = x.shape
        assert N == 2048
        out = torch.empty_like(x)
        sx0, sx1, sx2 = x.stride()
        so0, so1, so2 = out.stride()
        _rmsnorm_mul_kernel_3d[(D0 * D1,)](
            x,
            w,
            out,
            D0,
            D1,
            sx0,
            sx1,
            sx2,
            so0,
            so1,
            so2,
            1e-6,
            OUT_FP32=False,
            BLOCK_N=256,
            num_warps=4,
        )
        return out

    if route == "tinyllama_fp32":
        w, x = args[0], args[1]
        D0, D1, N = x.shape
        assert N == 2048
        out = torch.empty((D0, D1, N), device=x.device, dtype=torch.float32)
        sx0, sx1, sx2 = x.stride()
        so0, so1, so2 = out.stride()
        _rmsnorm_mul_kernel_3d[(D0 * D1,)](
            x,
            w,
            out,
            D0,
            D1,
            sx0,
            sx1,
            sx2,
            so0,
            so1,
            so2,
            1e-5,
            OUT_FP32=True,
            BLOCK_N=256,
            num_warps=4,
        )
        return out

    raise RuntimeError(f"Unknown route: {route}")