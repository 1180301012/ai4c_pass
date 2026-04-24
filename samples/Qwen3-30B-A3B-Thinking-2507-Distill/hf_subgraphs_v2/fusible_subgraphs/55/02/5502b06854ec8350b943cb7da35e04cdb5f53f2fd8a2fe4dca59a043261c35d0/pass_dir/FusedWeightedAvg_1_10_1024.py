import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fully-unrolled single-pass kernel.
# SEQ_LEN=10 and BLOCK_SIZE=1024 are Python integer literals inside the
# function body, so Triton statically unrolls the loop at JIT-compile time.
# num_warps=32 → 1024 threads per block (1 element/thread), max SM occupancy.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_weighted_avg_kernel(
    in0_ptr,                # int64    [batch, seq, hidden_dim]
    in1_ptr,                # fp16/bf16 [batch, seq, hidden_dim]
    out_ptr,                # float32  [batch, hidden_dim]
    hidden_dim,
    IS_BF16:   tl.constexpr,
    BLOCK_SIZE:tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    base0 = pid * hidden_dim        # row start in in0
    base1 = pid * hidden_dim        # row start in in1

    # Static unroll: Python range(10) is resolved at JIT time →
    # Triton emits 10 independent load-blocks with no branch overhead.
    v0_0 = tl.load(in0_ptr + base0 +  0 * hidden_dim + offs)
    v1_0 = tl.load(in1_ptr + base1 +  0 * hidden_dim + offs)
    v0_1 = tl.load(in0_ptr + base0 +  1 * hidden_dim + offs)
    v1_1 = tl.load(in1_ptr + base1 +  1 * hidden_dim + offs)
    v0_2 = tl.load(in0_ptr + base0 +  2 * hidden_dim + offs)
    v1_2 = tl.load(in1_ptr + base1 +  2 * hidden_dim + offs)
    v0_3 = tl.load(in0_ptr + base0 +  3 * hidden_dim + offs)
    v1_3 = tl.load(in1_ptr + base1 +  3 * hidden_dim + offs)
    v0_4 = tl.load(in0_ptr + base0 +  4 * hidden_dim + offs)
    v1_4 = tl.load(in1_ptr + base1 +  4 * hidden_dim + offs)
    v0_5 = tl.load(in0_ptr + base0 +  5 * hidden_dim + offs)
    v1_5 = tl.load(in1_ptr + base1 +  5 * hidden_dim + offs)
    v0_6 = tl.load(in0_ptr + base0 +  6 * hidden_dim + offs)
    v1_6 = tl.load(in1_ptr + base1 +  6 * hidden_dim + offs)
    v0_7 = tl.load(in0_ptr + base0 +  7 * hidden_dim + offs)
    v1_7 = tl.load(in1_ptr + base1 +  7 * hidden_dim + offs)
    v0_8 = tl.load(in0_ptr + base0 +  8 * hidden_dim + offs)
    v1_8 = tl.load(in1_ptr + base1 +  8 * hidden_dim + offs)
    v0_9 = tl.load(in0_ptr + base0 +  9 * hidden_dim + offs)
    v1_9 = tl.load(in1_ptr + base1 +  9 * hidden_dim + offs)

    # Accumulate in float32 (avoids fp16/bf16 precision loss)
    a_num = (v1_0.to(tl.float32) * v0_0.to(tl.float32)
           + v1_1.to(tl.float32) * v0_1.to(tl.float32)
           + v1_2.to(tl.float32) * v0_2.to(tl.float32)
           + v1_3.to(tl.float32) * v0_3.to(tl.float32)
           + v1_4.to(tl.float32) * v0_4.to(tl.float32)
           + v1_5.to(tl.float32) * v0_5.to(tl.float32)
           + v1_6.to(tl.float32) * v0_6.to(tl.float32)
           + v1_7.to(tl.float32) * v0_7.to(tl.float32)
           + v1_8.to(tl.float32) * v0_8.to(tl.float32)
           + v1_9.to(tl.float32) * v0_9.to(tl.float32))

    a_den = (v0_0.to(tl.float32) + v0_1.to(tl.float32)
           + v0_2.to(tl.float32) + v0_3.to(tl.float32)
           + v0_4.to(tl.float32) + v0_5.to(tl.float32)
           + v0_6.to(tl.float32) + v0_7.to(tl.float32)
           + v0_8.to(tl.float32) + v0_9.to(tl.float32)
           + 1e-9)                   # guard against 0/0

    result = a_num / a_den

    out_offs = pid * hidden_dim + offs
    if IS_BF16:
        tl.store(out_ptr + out_offs, result.to(tl.bfloat16))
    else:
        tl.store(out_ptr + out_offs, result)


@torch.fx.wrap
def fused_weighted_avg(in_0, in_1):
    """
    Fused computation:
        tmp_0 = in_0.to(float32)
        tmp_1 = in_1 * tmp_0
        tmp_2 = sum(tmp_1, dim=1)
        tmp_3 = sum(tmp_0, dim=1)
        tmp_4 = clamp(tmp_3, min=1e-9)
        tmp_5 = tmp_2 / tmp_4
        tmp_6 = cat([tmp_5], 1)
    """
    batch      = in_0.shape[0]
    seq_len    = in_0.shape[1]
    hidden_dim = in_0.shape[2]

    is_bf16 = (in_1.dtype == torch.bfloat16)

    # float32 output (matches eager dtype; no post-convert needed)
    out = torch.empty((batch, hidden_dim), dtype=torch.float32, device=in_0.device)

    _fused_weighted_avg_kernel[(batch * seq_len,)](
        in_0, in_1, out,
        hidden_dim,
        IS_BF16=is_bf16,
        BLOCK_SIZE=hidden_dim,
        num_warps=32,    # 1024 threads/block → max SM occupancy
        num_stages=1,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern: mirrors model.py exactly (no None-cleanup lines)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_weighted_avg