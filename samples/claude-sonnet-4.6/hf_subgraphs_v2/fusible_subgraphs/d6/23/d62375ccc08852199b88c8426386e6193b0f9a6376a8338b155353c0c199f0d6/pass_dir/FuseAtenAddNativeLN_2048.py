import torch
import operator
from pass_dir._shared_kernels import fused_dispatch


# ── Pass interface ────────────────────────────────────────────────────────────
# Pattern: aten.add.Tensor + aten.native_layer_norm (H=2048, decomposed graph)
# dropout(training=False) is removed during decomposition.
# native_layer_norm returns (output, mean, rstd); we extract index 0.
def pattern(x, y, weight, bias):
    add       = torch.ops.aten.add.Tensor(x, y)
    ln_result = torch.ops.aten.native_layer_norm.default(add, [2048], weight, bias, 1e-05)
    ln_out    = ln_result[0]
    return (add, ln_out)


def replacement_args(x, y, weight, bias):
    return (x, y, weight, bias, '2048')


def replacement_func():
    return fused_dispatch