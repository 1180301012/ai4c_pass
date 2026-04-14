import torch
from pass_dir.ln_kernels import triton_layernorm_dispatch


# ────────────────────────────────────────────────────────────
#  Pass interface  —  matches layer_norm( x, (16,), w, b, 1e-5 )
# ────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    tmp_4 = torch.nn.functional.layer_norm(in_2, (16,), in_1, in_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    # Append route string so the shared dispatch knows which kernel to call
    return (in_0, in_1, in_2, "n16")


def replacement_func():
    return triton_layernorm_dispatch