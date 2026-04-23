import torch
import triton
import triton.language as tl
from graph_net_bench.torch.backend import pass_mgr_backend as _pmb


if not hasattr(_pmb.PatternReplacementPass, "_ai4c_debug_graph_dump"):
    _orig_call = _pmb.PatternReplacementPass.__call__

    def _debug_call(self, gm: torch.fx.GraphModule):
        print("[AI4C DEBUG] FX graph for", self.pass_name, flush=True)
        print(gm.graph, flush=True)
        try:
            from graph_net_bench.torch.custom_replacement import force_args_symbolic_trace
            print("[AI4C DEBUG] Pattern graph for", self.pass_name, flush=True)
            print(force_args_symbolic_trace(self.pattern).graph, flush=True)
        except Exception as e:
            print("[AI4C DEBUG] Pattern graph dump failed:", e, flush=True)
        return _orig_call(self, gm)

    _pmb.PatternReplacementPass.__call__ = _debug_call
    _pmb.PatternReplacementPass._ai4c_debug_graph_dump = True


# Pattern matching function
# NOTE: Mirrors model.py exactly.
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7


# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_pool_bn_silu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    EPS: tl.constexpr,
):
    # One program handles one output channel.
    c = tl.program_id(0)

    # 8x8 output spatial positions.
    pos = tl.arange(0, 64)
    ox = pos & 7
    oy = pos >> 3

    # Input is logically reshaped to [1, 512, 16, 16].
    # Each channel owns a contiguous 16x16 = 256 element block.
    # For 2x2 avg pooling with stride 2:
    # input offset within channel = (2 * oy) * 16 + (2 * ox)
    in_base = c * 256 + (oy << 5) + (ox << 1)

    x0 = tl.load(x_ptr + in_base).to(tl.float32)
    x1 = tl.load(x_ptr + in_base + 1).to(tl.float32)
    x2 = tl.load(x_ptr + in_base + 16).to(tl.float32)
    x3 = tl.load(x_ptr + in_base + 17).to(tl.float32)
    pooled = (x0 + x1 + x2 + x3) * 0.25

    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    scale = weight * tl.rsqrt(var + EPS)
    shift = bias - mean * scale
    y = pooled * scale + shift

    sig = 1.0 / (1.0 + tl.exp(-y))
    y = y * sig

    out_offsets = c * 64 + pos
    tl.store(out_ptr + out_offsets, y)


# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_pool_bn_silu(mean, var, bias, weight, x):
    c = mean.numel()
    out = torch.empty((1, c, 8, 8), device=x.device, dtype=x.dtype)
    fused_pool_bn_silu_kernel[(c,)](
        x,
        mean,
        var,
        bias,
        weight,
        out,
        EPS=1e-5,
        num_warps=2,
        num_stages=1,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_pool_bn_silu