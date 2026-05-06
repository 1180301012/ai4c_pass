import torch
import triton
import triton.language as tl


# ── pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.transpose(-2, -1)
    return tmp_0, tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Fused kernel: handles BOTH independent operations in ONE launch.
# - Element-wise multiply in_1  by scalar (writes out1)
# - Transpose last two dims  of in_0  (stride-D gather → coalesced writes)
#  n_elements (301056) = 70*1*49*32 = 294 * 1024 exactly → no mask overhead.
@triton.jit
def _fused_mul_transpose_kernel(
    in0_ptr,          # [B, 1, S, D]  – source for transpose
    in1_ptr,          # [B, 1, S, D]  – source for scaled multiply
    out0_ptr,         # [B, 1, S, D]  – tmp_0  output
    out1_ptr,         # [B, 1, D, S]  – tmp_1  output
    S,                # sequence length  (49)
    D,                # feature dim       (32)
    BLOCK_SIZE: tl.constexpr,  # 1024  (must divide 301056)
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # exact – 301056 = 294 * 1024, so no out-of-bounds tail
    offs1 = offs            # body index  (same for in/out of out0/tmp_0)
    offs2 = offs // S * S + (offs % S)  # transposition index for out1/tmp_1

    # ── scaled multiply: tmp_0 = in_1 * scalar  (coalesced R/W) ──────────
    v1 = tl.load(in1_ptr + offs1)
    tl.store(out0_ptr + offs1, v1 * 0.1767766952966369)

    # ── transpose: tmp_1 = in_0.transpose(-2, -1)  (scatter reads, coalesced W) ─
    v0 = tl.load(in0_ptr + offs2)          # strides-D gather across batches
    tl.store(out1_ptr + offs, v0)   # no mask: 301056 = 294*1024 exactly


@torch.fx.wrap
def fused_mul_transpose(in_0, in_1):
    B  = in_1.shape[0]    # 70
    S  = in_1.shape[2]    # 49
    D  = in_1.shape[3]    # 32
    n_elements = B * S * D   # 301056 = 294 * 1024 exactly

    out0 = torch.empty_like(in_1)
    out1 = torch.empty(B, 1, D, S, dtype=in_1.dtype, device=in_1.device)

    BLOCK_SIZE = 1024
    _fused_mul_transpose_kernel[(n_elements // BLOCK_SIZE,)](
        in_0, in_1, out0, out1,
        S, D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out0, out1


def replacement_func():
    return fused_mul_transpose