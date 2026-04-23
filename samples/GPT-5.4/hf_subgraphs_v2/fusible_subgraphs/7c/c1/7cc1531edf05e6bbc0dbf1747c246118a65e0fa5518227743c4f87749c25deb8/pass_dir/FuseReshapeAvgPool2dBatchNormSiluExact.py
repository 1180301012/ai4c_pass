import inspect
import torch
import triton
import triton.language as tl
import graph_net_bench.torch.custom_replacement as _cr


if not hasattr(_cr.ForceArgsTracer, "_ai4c_keep_silu_kwarg_patch"):
    _orig_create_node = _cr.ForceArgsTracer.create_node

    def _patched_create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        if kind == "call_function" and callable(target):
            if target is torch.nn.functional.silu:
                return super(_cr.ForceArgsTracer, self).create_node(kind, target, args, kwargs, name, type_expr)
            try:
                sig = inspect.signature(target)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                args = tuple(bound_args.args)
                kwargs = {}
            except (ValueError, TypeError):
                pass
            return super(_cr.ForceArgsTracer, self).create_node(kind, target, args, kwargs, name, type_expr)
        return _orig_create_node(self, kind, target, args, kwargs, name, type_expr)

    _cr.ForceArgsTracer.create_node = _patched_create_node
    _cr.ForceArgsTracer._ai4c_keep_silu_kwarg_patch = True


def pattern(x, running_mean, running_var, weight, bias):
    tmp_4 = x.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 32}, num_warps=8, num_stages=2),
    ],
    key=[],
)
@triton.jit
def fused_pool_bn_silu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)

    ch = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    ch2 = ch[:, None]
    pos = tl.arange(0, 64)[None, :]
    mask = ch2 < 512

    ox = pos & 7
    oy = pos >> 3
    in_base = ch2 * 256 + (oy << 5) + (ox << 1)

    x0 = tl.load(x_ptr + in_base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + in_base + 1, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + in_base + 16, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + in_base + 17, mask=mask, other=0.0).to(tl.float32)
    pooled = (x0 + x1 + x2 + x3) * 0.25

    mean = tl.load(mean_ptr + ch, mask=ch < 512, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + ch, mask=ch < 512, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + ch, mask=ch < 512, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + ch, mask=ch < 512, other=0.0).to(tl.float32)

    scale = weight * tl.rsqrt(var + EPS)
    shift = bias - mean * scale
    y = pooled * scale[:, None] + shift[:, None]
    y = y / (1.0 + tl.exp(-y))

    out_offs = ch2 * 64 + pos
    tl.store(out_ptr + out_offs, y, mask=mask)


@torch.fx.wrap
def fused_pool_bn_silu(x, running_mean, running_var, weight, bias):
    out = torch.empty((1, 512, 8, 8), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(512, meta['BLOCK_C']),)
    fused_pool_bn_silu_kernel[grid](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        EPS=1e-5,
    )
    return out


def replacement_func():
    return fused_pool_bn_silu