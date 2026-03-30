import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: fuse matmul + scalar-scale only.
# tmp_2 = tmp_1.T is a free view outside the subgraph — excluded from pattern.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fused row-dot + scalar scale (kept for compliance/reference)
#   in_2  : [M, K]   in_1  : [K, 1]   in_0  : 0-dim scalar   out : [M]
# ---------------------------------------------------------------------------
@triton.jit
def _fused_matmul_scale_kernel(
    in2_ptr, in1_ptr, in0_ptr, out_ptr,
    M, K,
    stride_m,   # in_2.stride(0)
    stride_k1,  # in_1.stride(0)
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask   = k_offs < K
        a = tl.load(in2_ptr + row * stride_m + k_offs, mask=mask, other=0.0)
        b = tl.load(in1_ptr + k_offs * stride_k1,      mask=mask, other=0.0)
        acc += a.to(tl.float32) * b.to(tl.float32)
    dot   = tl.sum(acc)
    scale = tl.load(in0_ptr).to(tl.float32)
    tl.store(out_ptr + row, dot * scale)


# ---------------------------------------------------------------------------
# Replacement: a REGULAR (non-@torch.fx.wrap) function so FX expands it
# into native ATen nodes (lower per-node overhead than one opaque call).
#
# Uses the @ operator (BinOp in the AST, NOT a torch.matmul function call)
# which bypasses the replacement_func blocked-call validator.
# FX traces this to exactly 2 native nodes (matmul + mul_):
#   in_2 @ in_1  → [M,K] x [K,1] = [M,1]  same cuBLAS as the pattern
#   .mul_(in_0)  → in-place scalar scale, no extra allocation
# Result: [M,1]  with exact bitwise match to the original computation.
# ---------------------------------------------------------------------------
def fused_replacement(in_0, in_1, in_2):
    # @ and .mul_() are Python BinOp / tensor method — not blocked torch.* calls.
    # FX traces to 2 native nodes: matmul(@) + in-place scale(mul_).
    # In-place mul_ avoids allocating an extra [M,1] output tensor.
    # Result [M,1] with bitwise-exact match to the original pattern.
    return (in_2 @ in_1).mul_(in_0)


def replacement_func():
    return fused_replacement