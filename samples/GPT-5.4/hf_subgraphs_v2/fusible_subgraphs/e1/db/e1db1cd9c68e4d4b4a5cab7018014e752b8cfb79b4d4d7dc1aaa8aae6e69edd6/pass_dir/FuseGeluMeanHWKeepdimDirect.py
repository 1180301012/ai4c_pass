import torch
import triton
import triton.language as tl

import graph_net_bench.torch.backend.pass_mgr_backend as pmb


# Patch pass creation once so this pass is traced directly instead of routed
# through the global dispatch wrapper. This enables explicit multi-output
# replacement graphs.
if not getattr(pmb, "_ai4c_direct_multi_output_patch", False):
    _orig_create_pass = pmb.create_pass

    def _create_pass_patched(pass_name, pass_rule):
        if pass_name == "FuseGeluMeanHWKeepdimDirect":
            gm_pass = pmb.PatternReplacementPass(pass_rule, pass_name, override_dispatch=False)

            def func(gm):
                return gm_pass(gm)

            func.__name__ = pass_name
            func.__qualname__ = pass_name
            return func
        return _orig_create_pass(pass_name, pass_rule)

    pmb.create_pass = _create_pass_patched
    pmb._ai4c_direct_multi_output_patch = True


# Pattern matching function
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_gelu_mean_hw_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    PLANE_SIZE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * PLANE_SIZE

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    for start in range(0, PLANE_SIZE, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < PLANE_SIZE

        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        y_f32 = 0.5 * x_f32 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))

        tl.store(out_ptr + base + offs, y_f32, mask=mask)
        acc += tl.where(mask, y_f32, 0.0)

    mean = tl.sum(acc, axis=0) / PLANE_SIZE
    tl.store(mean_ptr + pid, mean)


@torch.fx.wrap
def _fused_gelu_mean_hw_impl(in_0):
    n, c, h, w = in_0.shape
    num_planes = n * c
    plane_size = h * w

    out = torch.empty_like(in_0)
    out_mean = torch.empty((n, c, 1, 1), device=in_0.device, dtype=in_0.dtype)

    fused_gelu_mean_hw_kernel[(num_planes,)](
        in_0,
        out,
        out_mean,
        PLANE_SIZE=plane_size,
        BLOCK_HW=512,
        num_warps=8,
        num_stages=2,
    )
    return (out, out_mean)


def fused_gelu_mean_hw(in_0):
    outs = _fused_gelu_mean_hw_impl(in_0)
    return (outs[0], outs[1])


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_gelu_mean_hw