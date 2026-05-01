import torch
from torch import device
from pass_dir.shared_fused_mask_softmax import fused_mask_softmax_dispatch


# ── Pattern ───────────────────────────────────────────────────────────────────
# clamp_val is a placeholder (wildcard) that matches the torch.tensor(-inf)
# node produced inside the model forward.  Making it a positional arg prevents
# the FX tracer from executing torch.tensor() eagerly and losing the node.

# ── Pattern ───────────────────────────────────────────────────────────────────
# Uses ATen-level ops matching what dynamo produces from the GraphModule.
# clamp_val is a placeholder matching the torch.tensor(-inf) constant node.
# dropout(training=False) is an identity and is eliminated by dynamo.

def pattern(in_0, in_1, clamp_val):
    tmp_0 = in_1 + in_0
    tmp_2 = torch.max(tmp_0, clamp_val)
    tmp_3 = tmp_2.view(16, 13, 13)
    return tmp_3


def replacement_args(in_0, in_1, clamp_val):
    return (in_0, in_1, "16_13_13")


def replacement_func():
    return fused_mask_softmax_dispatch