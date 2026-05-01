import torch
from pass_dir._swin_shared import swin_fused_dispatch


# ─── Pattern ──────────────────────────────────────────────────────────────────
# Match layer_norm + dropout only. The dropout is a no-op (p=0, training=False)
# so fusing them eliminates one kernel launch while handling the non-contiguous
# transposed input. The view/pad/permute chain runs normally on the result.
def pattern(tmp_7, in_2, in_1):
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (96,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


# ─── Replacement args ─────────────────────────────────────────────────────────
def replacement_args(tmp_7, in_2, in_1):
    return (tmp_7, in_2, in_1, "c96")


# ─── Replacement func — returns the SAME shared dispatch object as C16 pass ──
def replacement_func():
    return swin_fused_dispatch