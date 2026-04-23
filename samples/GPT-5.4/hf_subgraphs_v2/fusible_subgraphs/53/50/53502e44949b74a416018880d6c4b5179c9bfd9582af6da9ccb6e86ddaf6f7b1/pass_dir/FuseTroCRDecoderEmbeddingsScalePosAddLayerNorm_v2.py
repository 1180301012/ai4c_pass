import torch
import triton
import triton.language as tl


HIDDEN = 256
EPS = 1e-5



def _maybe_print_pattern_graph_once():
    if globals().get("_AI4C_V2_PATTERN_PRINTED", False):
        return
    globals()["_AI4C_V2_PATTERN_PRINTED"] = True
    try:
        from graph_net_bench.torch.custom_replacement import force_args_symbolic_trace
        print("[AI4C DEBUG V2] PATTERN GRAPH START", flush=True)
        print(force_args_symbolic_trace(pattern).graph, flush=True)
        print("[AI4C DEBUG V2] PATTERN GRAPH END", flush=True)
    except Exception as e:
        print(f"[AI4C DEBUG V2] PATTERN GRAPH TRACE FAILED: {e}", flush=True)


def pattern(pos_ids, pos_weight, tok_weight, ln_bias, ln_weight, input_ids):
    tok = torch.nn.functional.embedding(input_ids, tok_weight, 1, None, 2.0, False, False)
    tok_scaled = tok * 16.0
    pos = torch.nn.functional.embedding(pos_ids, pos_weight, None, None, 2.0, False, False)
    added = tok_scaled + pos
    normed = torch.nn.functional.layer_norm(added, (256,), ln_weight, ln_bias, 1e-05)
    out = torch.nn.functional.dropout(normed, p=0.1, training=False)
    return out


_maybe_print_pattern_graph_once()


def replacement_args(pos_ids, pos_weight, tok_weight, ln_bias, ln_weight, input_ids):
    return (pos_weight, tok_weight, ln_bias, ln_weight, input_ids, pos_ids)


@triton.jit
def _fused_trocr_embed_ln_kernel(
    pos_weight_ptr,
    tok_weight_ptr,
    ln_bias_ptr,
    ln_weight_ptr,
    input_ids_ptr,
    pos_ids_ptr,
    out_ptr,
    HIDDEN_SIZE: tl.constexpr,
    EPS_VALUE: tl.constexpr,
):
    offs = tl.arange(0, HIDDEN_SIZE)

    token_idx = tl.load(input_ids_ptr)
    pos_idx = tl.load(pos_ids_ptr)

    tok = tl.load(tok_weight_ptr + token_idx * HIDDEN_SIZE + offs).to(tl.float32)
    pos = tl.load(pos_weight_ptr + pos_idx * HIDDEN_SIZE + offs).to(tl.float32)
    x = tok * 16.0 + pos

    mean = tl.sum(x, axis=0) * (1.0 / HIDDEN_SIZE)
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) * (1.0 / HIDDEN_SIZE)
    inv_std = tl.rsqrt(var + EPS_VALUE)

    gamma = tl.load(ln_weight_ptr + offs).to(tl.float32)
    beta = tl.load(ln_bias_ptr + offs).to(tl.float32)
    y = x_centered * inv_std * gamma + beta

    tl.store(out_ptr + offs, y)


@torch.fx.wrap
def fused_trocr_embed_ln(pos_weight, tok_weight, ln_bias, ln_weight, input_ids, pos_ids):
    out = torch.empty((1, 1, HIDDEN), device=tok_weight.device, dtype=tok_weight.dtype)
    _fused_trocr_embed_ln_kernel[(1,)](
        pos_weight,
        tok_weight,
        ln_bias,
        ln_weight,
        input_ids,
        pos_ids,
        out,
        HIDDEN_SIZE=HIDDEN,
        EPS_VALUE=EPS,
        num_warps=2,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_trocr_embed_ln