import torch
from pass_dir.shared_fused_ln import dispatch_fused_add_layernorm


# ---------------------------------------------------------------------------
# Pattern: add two hidden-state tensors, then layer-normalize over last dim
# normalized_shape = (16,), eps = 1e-05
# Matches bfloat16 and float16 models (tiny D2Vec)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (16,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    # Pass route string so the shared dispatch knows which variant to select
    return (in_0, in_1, in_2, in_3, "16")


def replacement_func():
    return dispatch_fused_add_layernorm