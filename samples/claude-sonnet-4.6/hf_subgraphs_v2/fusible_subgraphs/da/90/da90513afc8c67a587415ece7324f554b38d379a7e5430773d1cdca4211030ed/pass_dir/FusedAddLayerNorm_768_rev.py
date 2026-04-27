import torch
import triton
import triton.language as tl
from pass_dir.kernel_impl import fused_add_layernorm_dispatch


# ── Pattern ──────────────────────────────────────────────────────────────────
# float32 graph has operands reversed: in_3 + in_2
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3


# ── Argument extractor  (appends route string) ───────────────────────────────
def replacement_args(in_0, in_1, in_2, in_3):
    # x=in_3, y=in_2, weight=in_1, bias=in_0, route="N768"
    return (in_3, in_2, in_1, in_0, "N768")


# ── Shared replacement function ──────────────────────────────────────────────
def replacement_func():
    return fused_add_layernorm_dispatch