import torch
from pass_dir.dispatch_kernels import dispatch_kernel


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: torch.matmul(in_1, in_0) → view(B, C1, 1, 1)
#   in_0 : [B, 1, K, 1]       (N=1 → GEMV / batched mat-vec)
#   in_1 : [B, 1, C1, K]
#   out  : [B, C1, 1, 1]
#   Used by: GCNet-R101, S-ViPNAS-Res50
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    # Matches:  tmp_0 = torch.matmul(in_1, in_0)
    tmp_0 = torch.matmul(in_1, in_0)
    return tmp_0


def replacement_args(in_0, in_1):
    # Append route string so dispatch_kernel knows which kernel to call.
    return (in_0, in_1, "route_1x1")


def replacement_func():
    # Return the SAME shared dispatch_kernel object (satisfies replacement_func_limit)
    from pass_dir.dispatch_kernels import dispatch_kernel
    return dispatch_kernel