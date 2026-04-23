import torch
import triton
import triton.language as tl

from torch import device

import graph_net_bench.torch.backend.pass_mgr_backend as _pmb


if not hasattr(_pmb, "_ai4c_debug_gm_print_patch"):
    _orig_call = _pmb.PatternReplacementPass.__call__

    def _debug_call(self, gm):
        print("[AI4C DEBUG] GM GRAPH START", flush=True)
        print(gm.graph, flush=True)
        print("[AI4C DEBUG] GM GRAPH END", flush=True)
        return _orig_call(self, gm)

    _pmb.PatternReplacementPass.__call__ = _debug_call
    _pmb._ai4c_debug_gm_print_patch = True



HIDDEN = 256
POS_INDEX = 2
SCALE = 16.0
EPS = 1e-5


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = torch.nn.functional.embedding(in_4, in_1, 1, None, 2.0, False, False)
    tmp_5 = tmp_4 * 16.0
    tmp_6 = torch.arange(0, 1, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_7 = tmp_6.expand(1, -1)
    tmp_8 = tmp_7 + 2
    tmp_9 = torch.nn.functional.embedding(tmp_8, in_0, None, None, 2.0, False, False)
    tmp_10 = tmp_5 + tmp_9
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_3, in_2, 1e-05)
    tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.1, training=False)
    return tmp_12


def _maybe_print_pattern_graph_once():
    if globals().get("_AI4C_PATTERN_PRINTED", False):
        return
    globals()["_AI4C_PATTERN_PRINTED"] = True
    try:
        from graph_net_bench.torch.custom_replacement import force_args_symbolic_trace
        print("[AI4C DEBUG] PATTERN GRAPH START", flush=True)
        print(force_args_symbolic_trace(pattern).graph, flush=True)
        print("[AI4C DEBUG] PATTERN GRAPH END", flush=True)
    except Exception as e:
        print(f"[AI4C DEBUG] PATTERN GRAPH TRACE FAILED: {e}", flush=True)


_maybe_print_pattern_graph_once()



def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def _fused_trocr_embed_ln_kernel(
    pos_weight_ptr,
    tok_weight_ptr,
    ln_bias_ptr,
    ln_weight_ptr,
    input_ids_ptr,
    out_ptr,
    HIDDEN_SIZE: tl.constexpr,
    POS_ROW: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    EPS_VALUE: tl.constexpr,
):
    offs = tl.arange(0, HIDDEN_SIZE)
    token_idx = tl.load(input_ids_ptr)

    tok = tl.load(tok_weight_ptr + token_idx * HIDDEN_SIZE + offs).to(tl.float32)
    pos = tl.load(pos_weight_ptr + POS_ROW * HIDDEN_SIZE + offs).to(tl.float32)
    x = tok * SCALE_VALUE + pos

    mean = tl.sum(x, axis=0) * (1.0 / HIDDEN_SIZE)
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) * (1.0 / HIDDEN_SIZE)
    inv_std = tl.rsqrt(var + EPS_VALUE)

    ln_weight = tl.load(ln_weight_ptr + offs).to(tl.float32)
    ln_bias = tl.load(ln_bias_ptr + offs).to(tl.float32)
    y = x_centered * inv_std * ln_weight + ln_bias

    tl.store(out_ptr + offs, y)


@torch.fx.wrap
def fused_trocr_embed_ln(in_0, in_1, in_2, in_3, in_4):
    out = torch.empty((1, 1, HIDDEN), device=in_1.device, dtype=in_1.dtype)
    _fused_trocr_embed_ln_kernel[(1,)](
        in_0,
        in_1,
        in_2,
        in_3,
        in_4,
        out,
        HIDDEN_SIZE=HIDDEN,
        POS_ROW=POS_INDEX,
        SCALE_VALUE=SCALE,
        EPS_VALUE=EPS,
        num_warps=4,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_trocr_embed_ln