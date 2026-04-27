import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Single Triton kernel — no autotune, all meta-params passed at launch time
# 3-D grid: dim-0=HW_tiles, dim-1=C_out_tiles, dim-2=N_batch
# ---------------------------------------------------------------------------
@triton.jit
def _conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N_batch, C_in, HW, C_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in range(0, tl.cdiv(C_in, BLOCK_K)):
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        w = tl.load(
            weight_ptr + offs_m[:, None] * C_in + offs_k[None, :],
            mask=(offs_m[:, None] < C_out) & (offs_k[None, :] < C_in),
            other=0.0, eviction_policy='evict_last',
        )
        x = tl.load(
            input_ptr + pid_b * C_in * HW + offs_k[:, None] * HW + offs_n[None, :],
            mask=(offs_k[:, None] < C_in) & (offs_n[None, :] < HW),
            other=0.0,
        )
        acc += tl.dot(w, x, allow_tf32=True)
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < C_out, other=0.0)
    acc += bias[:, None].to(tl.float32)
    out_mask = (offs_m[:, None] < C_out) & (offs_n[None, :] < HW)
    tl.store(
        output_ptr + pid_b * C_out * HW + offs_m[:, None] * HW + offs_n[None, :],
        acc.to(output_ptr.dtype.element_ty),
        mask=out_mask, eviction_policy='evict_first',
    )


# ---------------------------------------------------------------------------
# Wrapper with hardcoded config selection (no autotune instability)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def conv1x1_bias_reshape(bias, weight, input_tensor):
    N_batch = input_tensor.shape[0]
    C_in    = input_tensor.shape[1]
    H       = input_tensor.shape[2]
    W       = input_tensor.shape[3]
    HW      = H * W
    C_out   = weight.shape[0]
    ESIZE   = input_tensor.element_size()   # 2 = bf16/fp16, 4 = fp32

    output = torch.empty((N_batch, C_out, HW),
                         dtype=input_tensor.dtype,
                         device=input_tensor.device)

    # ── Config selection table (shared-memory budgets verified) ─────────────────
    # bf16/fp16 (ESIZE=2):
    #   Large  N≥64 : BLOCK_N=512, BK=32, ns=3 → 3×(2+32)=102 KB ✓ → best L2 reuse
    #   Medium N=8-63: BLOCK_N=256, BK=64, ns=3 → 3×(4+32)=108 KB ✓
    #   Small  N<8  : BLOCK_N=128, BK=64, ns=4 → 4×(2+16)=72 KB  ✓
    # float32 (ESIZE=4):
    #   Large  N≥64 : BLOCK_N=512, BK=32, ns=1 → 1×(4+64)=68 KB  ✓
    #   Medium N=8-63: BLOCK_N=256, BK=32, ns=3 → 3×(4+32)=108 KB ✓
    #   Small  N<8  : BLOCK_N=128, BK=64, ns=3 → 3×(8+32)=120 KB ✓
    if ESIZE <= 2:
        if N_batch >= 64:
            BM, BN, BK, NW, NS = 32, 512, 32, 8, 3
        elif N_batch >= 8:
            BM, BN, BK, NW, NS = 32, 256, 64, 8, 3
        else:
            BM, BN, BK, NW, NS = 32, 128, 64, 8, 4
    else:
        if N_batch >= 64:
            BM, BN, BK, NW, NS = 32, 512, 32, 8, 1
        elif N_batch >= 8:
            BM, BN, BK, NW, NS = 32, 256, 32, 8, 3
        else:
            BM, BN, BK, NW, NS = 32, 128, 64, 8, 3

    grid = (
        triton.cdiv(HW,    BN),
        triton.cdiv(C_out, BM),
        N_batch,
    )

    _conv1x1_kernel[grid](
        input_tensor, weight, bias, output,
        N_batch, C_in, HW, C_out,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_warps=NW, num_stages=NS,
    )

    return output


# ---------------------------------------------------------------------------
# Replacement entry point
# ---------------------------------------------------------------------------
def replacement_func():
    return conv1x1_bias_reshape