import operator
import torch
import torch.fx.proxy
import triton
import triton.language as tl
# Both subgraph patterns share this same function object → satisfies
# output_pass_replacement_func_limit == 1.
try:
    from pass_dir.kernel_impl import shared_dispatch
except ImportError:
    from kernel_impl import shared_dispatch


# ── 1. Monkey-patch Proxy.__iadd__ ────────────────────────────────────────────
def _proxy_iadd(self, other):
    return self.tracer.create_proxy('call_function', operator.iadd, (self, other), {})

torch.fx.proxy.Proxy.__iadd__ = _proxy_iadd


# ── 2. Pattern: iadd + iadd (do NOT include relu) ────────────────────────────
# Dynamo captures `in_3 += in_0; in_4 += in_2` as two iadd call_function nodes.
# ForceArgsTracer creates two iadd nodes whose target == Dynamo target == iadd.
# Matching ONLY these two nodes avoids the ForceArgsTracer kwarg-vs-positional
# issue that plagues the relu(iadd2, inplace=True) call.
# After replacement the downstream Dynamo-compiled relu applies on top of the
# kernel output → semantically correct.
def pattern(in_0, in_2, in_3):
    in_3 += in_0
    in_3 += in_2
    return in_3


# ── 3. Replacement args ───────────────────────────────────────────────────────
def replacement_args(in_0, in_2, in_3):
    # Route "add_add" → fused sum of all three inputs via Triton (no relu,
    # let the downstream Dynamo relu node apply).
    return (in_0, in_2, in_3, "add_add")


# ── 4. Shared dispatch (same object → output_pass_replacement_func limit == 1) ─
def replacement_func():
    return shared_dispatch