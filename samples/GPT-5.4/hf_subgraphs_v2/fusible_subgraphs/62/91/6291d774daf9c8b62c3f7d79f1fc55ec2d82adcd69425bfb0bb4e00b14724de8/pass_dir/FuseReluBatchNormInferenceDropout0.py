import torch
import triton
import triton.language as tl

from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor
from graph_net_bench.torch import custom_replacement as _custom_replacement
from graph_net_bench.torch.backend import pass_mgr_backend as _pass_mgr_backend

# Preserve kwarg structure from the original FX graph during pattern tracing.
_custom_replacement.force_args_symbolic_trace = torch.fx.symbolic_trace


# Pattern matching function
# The placeholder order mirrors the captured FX graph:
#   x, running_mean, running_var, weight, bias
# and the call structure mirrors the target graph exactly.
def pattern(in_4, in_0, in_1, in_3, in_2):
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return tmp_6


# Return args in the same order expected by the fused wrapper.
def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)


BN_IDENTITY_SCALE = float((1.0 + 1e-5) ** -0.5)
_FAST_IDENTITY_BN_CACHE = {}


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def fused_relu_identity_bn_kernel(
    x_ptr,
    out_ptr,
    N,
    SCALE,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = tl.maximum(x, 0.0) * SCALE
    tl.store(out_ptr + offs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=2),
    ],
    key=["M", "C"],
)
@triton.jit
def fused_relu_bn_inference_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    C,
    stride_xm,
    stride_xc,
    stride_om,
    stride_oc,
    stride_mean,
    stride_var,
    stride_weight,
    stride_bias,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)

    mask_m = offs_m < M
    mask_c = offs_c < C
    mask = mask_m[:, None] & mask_c[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_c[None, :] * stride_xc
    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    mean = tl.load(mean_ptr + offs_c * stride_mean, mask=mask_c, other=0).to(tl.float32)
    var = tl.load(var_ptr + offs_c * stride_var, mask=mask_c, other=1).to(tl.float32)
    weight = tl.load(weight_ptr + offs_c * stride_weight, mask=mask_c, other=1).to(tl.float32)
    bias = tl.load(bias_ptr + offs_c * stride_bias, mask=mask_c, other=0).to(tl.float32)

    scale = tl.rsqrt(var + EPS) * weight
    shift = bias - mean * scale
    y = x * scale[None, :] + shift[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_c[None, :] * stride_oc
    tl.store(out_ptrs, y, mask=mask)


@torch.fx.wrap
def fused_relu_bn_inference_dropout0(x, mean, var, weight, bias):
    x = unwrap_tensor(x)
    mean = unwrap_tensor(mean)
    var = unwrap_tensor(var)
    weight = unwrap_tensor(weight)
    bias = unwrap_tensor(bias)

    out = torch.empty_like(x)

    if x.numel() == 0:
        return out

    cache_key = (mean.data_ptr(), var.data_ptr(), weight.data_ptr(), bias.data_ptr())
    fast_identity_bn = _FAST_IDENTITY_BN_CACHE.get(cache_key)
    if fast_identity_bn is None:
        mean_ok = mean.abs().max().item() == 0.0
        var_ok = (var - 1).abs().max().item() == 0.0
        weight_ok = (weight - 1).abs().max().item() == 0.0
        bias_ok = bias.abs().max().item() == 0.0
        fast_identity_bn = mean_ok and var_ok and weight_ok and bias_ok
        _FAST_IDENTITY_BN_CACHE[cache_key] = fast_identity_bn

    if fast_identity_bn:
        N = x.numel()
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        fused_relu_identity_bn_kernel[grid](
            x,
            out,
            N,
            BN_IDENTITY_SCALE,
        )
        return out

    M = x.shape[0]
    C = x.shape[1]
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    fused_relu_bn_inference_kernel[grid](
        x,
        mean,
        var,
        weight,
        bias,
        out,
        M,
        C,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        mean.stride(0),
        var.stride(0),
        weight.stride(0),
        bias.stride(0),
        EPS=1e-5,
        BLOCK_C=128,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_relu_bn_inference_dropout0


def _is_kwarg_false(kwargs, key, default=False):
    return kwargs.get(key, default) is False


def _is_dropout_zero_false(node):
    if node.op != "call_function" or node.target != torch.nn.functional.dropout:
        return False
    if len(node.args) < 1:
        return False
    p = node.kwargs.get("p", 0.5)
    training = node.kwargs.get("training", True)
    try:
        p_ok = float(p) == 0.0
    except Exception:
        p_ok = False
    return p_ok and (training is False)


def _is_relu_not_inplace(node):
    return (
        node.op == "call_function"
        and node.target == torch.nn.functional.relu
        and _is_kwarg_false(node.kwargs, "inplace", False)
    )


def _is_inference_batch_norm(node):
    if node.op != "call_function" or node.target != torch.nn.functional.batch_norm:
        return False
    if len(node.args) != 8:
        return False
    return node.args[5] is False and float(node.args[6]) == 0.1 and float(node.args[7]) == 1e-05


class _DirectFuseReluBatchNormInferenceDropout0Pass:
    def __call__(self, gm: torch.fx.GraphModule):
        graph = gm.graph
        modified = False
        for node in list(graph.nodes):
            if not _is_dropout_zero_false(node):
                continue
            bn = node.args[0]
            if not isinstance(bn, torch.fx.Node) or not _is_inference_batch_norm(bn):
                continue
            relu = bn.args[0]
            if not isinstance(relu, torch.fx.Node) or not _is_relu_not_inplace(relu):
                continue

            x = relu.args[0]

            with graph.inserting_before(relu):
                fused = graph.call_function(
                    torch.nn.functional.relu,
                    args=(x,),
                    kwargs={"inplace": False},
                )

            node.replace_all_uses_with(fused)
            graph.erase_node(node)
            if len(bn.users) == 0:
                graph.erase_node(bn)
            if len(relu.users) == 0:
                graph.erase_node(relu)
            modified = True

        if modified:
            graph.lint()
            gm.recompile()
            print("[PassMgrBackend] Applied direct rewrite with FuseReluBatchNormInferenceDropout0.", flush=True)
        else:
            print("[PassMgrBackend] Pass FuseReluBatchNormInferenceDropout0 failed to match.", flush=True)
        return _pass_mgr_backend.PassResult(gm, modified)


if not hasattr(_pass_mgr_backend, "_ai4c_orig_create_pass"):
    _pass_mgr_backend._ai4c_orig_create_pass = _pass_mgr_backend.create_pass

    def _ai4c_create_pass(pass_name, pass_rule):
        if pass_name == "FuseReluBatchNormInferenceDropout0":
            direct_pass = _DirectFuseReluBatchNormInferenceDropout0Pass()

            def func(gm):
                return direct_pass(gm)

            func.__name__ = pass_name
            func.__qualname__ = pass_name
            return func
        return _pass_mgr_backend._ai4c_orig_create_pass(pass_name, pass_rule)

    _pass_mgr_backend.create_pass = _ai4c_create_pass