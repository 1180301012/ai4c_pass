import torch
from pass_dir.shared_fused_ln import dispatch_fused_add_layernorm


# ---------------------------------------------------------------------------
# Pattern: add two hidden-state tensors, then layer-normalize over last dim
# normalized_shape = (1024,), eps = 1e-05
# Matches float16 models
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    # Pass route string so the shared dispatch knows which variant to select
    return (in_0, in_1, in_2, in_3, "1024")


def replacement_func():
    return dispatch_fused_add_layernorm