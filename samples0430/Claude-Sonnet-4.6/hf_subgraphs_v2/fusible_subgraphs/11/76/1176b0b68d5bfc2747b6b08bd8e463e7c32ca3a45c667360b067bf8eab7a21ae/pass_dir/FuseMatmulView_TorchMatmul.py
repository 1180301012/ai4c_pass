import torch
from pass_dir._shared_gemm import dispatch_matmul  # noqa: F401


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: torch.matmul(in_1, in_0)  (GCNet, S-ViPNAS, float32 YOLO files)
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    result = torch.matmul(in_1, in_0)
    return result


def replacement_args(in_0, in_1):
    return (in_0, in_1, "mm")


# ──────────────────────────────────────────────────────────────────────────────
# Required by the AI4C framework.
# Returns the SAME dispatch_matmul object as FuseMatmulView.py so both passes
# share one unique replacement_func and bypass output_pass_replacement_func_limit.
# ──────────────────────────────────────────────────────────────────────────────

def replacement_func():
    return dispatch_matmul