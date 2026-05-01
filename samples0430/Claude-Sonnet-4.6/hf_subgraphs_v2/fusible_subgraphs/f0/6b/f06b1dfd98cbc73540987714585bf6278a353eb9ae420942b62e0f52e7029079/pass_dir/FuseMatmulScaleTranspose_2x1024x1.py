import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: fuse matmul([2,1024], [1024,1]) + scalar-scale into one kernel.
# Returns only tmp_1 (single output) so the framework's 1-output replacement
# contract is satisfied. tmp_2 = tmp_1.t() stays in the graph unchanged.
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel – specialised for M=2, N=1024
#   grid = (2,): one CTA per output row
#   BLOCK_N == N == 1024, so no masking needed
#   fp32 accumulation for numerical correctness
# ---------------------------------------------------------------------------

@triton.jit
def _fused_matmul_scale_kernel(
    in0_ptr,                # 0-dim scalar tensor (logit_scale)
    in1_ptr,                # [1024, 1]  C-contiguous
    in2_ptr,                # [2, 1024]  C-contiguous
    out_ptr,                # [2, 1]     C-contiguous
    BLOCK_N: tl.constexpr,  # == 1024
):
    row = tl.program_id(0)                     # 0 or 1
    offs = tl.arange(0, BLOCK_N)

    # No mask: BLOCK_N == N, all offsets are valid
    a = tl.load(in2_ptr + row * BLOCK_N + offs)
    b = tl.load(in1_ptr + offs)

    dot    = tl.sum(a.to(tl.float32) * b.to(tl.float32), axis=0)
    scale  = tl.load(in0_ptr).to(tl.float32)
    result = (dot * scale).to(a.dtype)

    tl.store(out_ptr + row, result)            # out[row, 0] at offset = row


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    # Shapes are fixed: in_2=[2,1024], in_1=[1024,1], in_0=scalar
    out = torch.empty((2, 1), dtype=in_2.dtype, device=in_2.device)

    _fused_matmul_scale_kernel[(2,)](
        in_0, in_1, in_2, out,
        BLOCK_N=1024,
        num_warps=4,
        num_stages=1,
    )

    return out


def replacement_func():
    return fused_matmul_scale