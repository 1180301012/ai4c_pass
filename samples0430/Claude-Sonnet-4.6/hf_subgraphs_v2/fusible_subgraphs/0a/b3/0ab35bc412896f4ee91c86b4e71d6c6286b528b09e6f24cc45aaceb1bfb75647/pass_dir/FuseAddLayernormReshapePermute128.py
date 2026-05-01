import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: add + layer_norm + reshape + permute + contiguous + permute + reshape
# The reshape/permute/contiguous/permute/reshape sequence is a mathematical
# identity (output[0,i,j] == layer_norm_out[0,i,j], same shape [1,4,128]).
# We fuse everything into a single Triton kernel.
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)
    tmp_4 = tmp_3.reshape(1, 2, 2, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.permute(0, 2, 3, 1)
    tmp_8 = tmp_7.reshape(1, -1, 128)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel: fused residual-add + layer_norm
#
# One program per row → the GPU can run all 4 blocks concurrently on
# separate SMs.  num_warps=1 (32 threads for BLOCK_SIZE=128) avoids
# inter-warp synchronisation in the reduction.
# Single-pass stats (E[x²]-E[x]²) avoids a second pass over the data.
# ---------------------------------------------------------------------------

@triton.jit
def fused_add_layernorm_128_kernel(
    in2_ptr,        # [N_ROWS, BLOCK_SIZE] input A
    in3_ptr,        # [N_ROWS, BLOCK_SIZE] input B (added to A)
    weight_ptr,     # [BLOCK_SIZE] layer-norm weight
    bias_ptr,       # [BLOCK_SIZE] layer-norm bias
    out_ptr,        # [N_ROWS, BLOCK_SIZE] output  (fp16 or bf16)
    BLOCK_SIZE: tl.constexpr,   # 128
    IS_BF16: tl.constexpr,      # True → bfloat16, False → float16
):
    row_idx = tl.program_id(0)
    col     = tl.arange(0, BLOCK_SIZE)
    base    = row_idx * BLOCK_SIZE

    # Issue ALL loads early so L2→register transfers of w/b can overlap
    # with the subsequent reduction operations (latency hiding).
    x2 = tl.load(in2_ptr    + base + col).to(tl.float32)
    x3 = tl.load(in3_ptr    + base + col).to(tl.float32)
    w  = tl.load(weight_ptr +        col).to(tl.float32)
    b  = tl.load(bias_ptr   +        col).to(tl.float32)

    # Residual add
    x = x2 + x3

    # Single-pass statistics: E[x] and E[x²] in one data pass
    x_sq    = x * x
    inv_N   = 1.0 / BLOCK_SIZE
    mean    = tl.sum(x,    axis=0) * inv_N
    mean_sq = tl.sum(x_sq, axis=0) * inv_N
    # var = E[x²] − (E[x])²  — clamp to 0 for numerical safety
    var     = tl.maximum(mean_sq - mean * mean, 0.0)

    # Normalize + affine  (w/b already in registers; rsqrt = single HW instr)
    rstd = tl.rsqrt(var + 1e-5)
    out  = (x - mean) * rstd * w + b

    # Store in original dtype
    if IS_BF16:
        tl.store(out_ptr + base + col, out.to(tl.bfloat16))
    else:
        tl.store(out_ptr + base + col, out.to(tl.float16))


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_layernorm_128(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [128]
    in_1 : weight [128]
    in_2 : [1, 4, 128]
    in_3 : [1, 4, 128]
    Returns: [1, 4, 128]
    """
    B, S, N = in_2.shape   # 1, 4, 128
    rows = B * S            # 4

    out = torch.empty_like(in_2)

    is_bf16 = in_2.dtype == torch.bfloat16

    # grid=(4,): one block per row, GPU runs all 4 concurrently.
    # num_warps=1: 32 threads × 4 elements each = pure intra-warp reduction,
    # zero inter-warp sync — optimal for BLOCK_SIZE=128.
    fused_add_layernorm_128_kernel[(rows,)](
        in_2, in_3, in_1, in_0, out,
        BLOCK_SIZE=128,
        IS_BF16=is_bf16,
        num_warps=1,
        num_stages=1,
    )

    return out


def replacement_func():
    return fused_add_layernorm_128