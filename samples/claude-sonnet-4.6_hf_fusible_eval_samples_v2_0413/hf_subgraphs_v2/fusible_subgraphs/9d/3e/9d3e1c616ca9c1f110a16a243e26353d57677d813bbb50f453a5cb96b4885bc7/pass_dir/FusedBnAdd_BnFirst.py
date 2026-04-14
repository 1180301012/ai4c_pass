"""
Pass: FusedBnAdd_BnFirst

Matches the inference BatchNorm subgraph and replaces it with a
Triton-fused BN kernel. Conv and residual-add remain in the graph
(handled by PyTorch).  Targeting a simple anchor: F.batch_norm.
"""

import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import dispatch_conv_bn_add


# ---------------------------------------------------------------------------
# Triton BN-only kernel (no conv, no add)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["HW"],
)
@triton.jit
def _bn_infer_kernel(
    x_ptr, mean_ptr, var_ptr, w_ptr, b_ptr, out_ptr,
    C, HW, eps,
    BLOCK_SIZE: tl.constexpr,
):
    nc_pid = tl.program_id(0)
    hw_pid = tl.program_id(1)
    c_idx  = nc_pid % C
    mean   = tl.load(mean_ptr + c_idx).to(tl.float32)
    var    = tl.load(var_ptr  + c_idx).to(tl.float32)
    w      = tl.load(w_ptr    + c_idx).to(tl.float32)
    b      = tl.load(b_ptr    + c_idx).to(tl.float32)
    scale  = w / tl.sqrt(var + eps)
    shift  = b - mean * scale
    base     = nc_pid * HW
    offsets  = hw_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < HW
    x        = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    result   = x * scale + shift
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + base + offsets, result.to(out_dtype), mask=mask)


@torch.fx.wrap
def _triton_bn(x, mean, var, bn_w, bn_b):
    N, C, H, W = x.shape
    HW  = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (N * C, triton.cdiv(HW, meta["BLOCK_SIZE"]))
    _bn_infer_kernel[grid](x, mean, var, bn_w, bn_b, out, C, HW, 1e-5)
    return out


# ---------------------------------------------------------------------------
# Pattern: just the batch_norm (anchor = F.batch_norm node)
# ---------------------------------------------------------------------------

def pattern(x, running_mean, running_var, bn_weight, bn_bias):
    """
    Matches any inference batch_norm call in the graph.
    Conv (upstream) and add (downstream) stay in the original graph.
    """
    bn_out = torch.nn.functional.batch_norm(
        x, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05
    )
    return bn_out


def replacement_args(x, running_mean, running_var, bn_weight, bn_bias):
    return (x, running_mean, running_var, bn_weight, bn_bias)


def replacement_func():
    return _triton_bn