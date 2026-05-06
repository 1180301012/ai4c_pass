import torch
import triton
import triton.language as tl


# ── Triton kernel: float16 ─────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['C_total'],
)
@triton.jit
def _pool_cat_fp16(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C_total: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Each program handles one (batch, channel) pair."""
    pid = tl.program_id(0)
    b = pid // C_total
    c = pid % C_total

    hw_offs = tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW

    base_in0 = b * C0 * HW
    base_in1 = b * C1 * HW
    base_in2 = b * C2 * HW
    base_in3 = b * C3 * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    if c < C0:
        ptr = in0_ptr + base_in0 + c * HW + hw_offs
        val = tl.load(ptr, mask=hw_mask, other=0.0).to(tl.float32)
        acc = acc + val

    elif c < C0 + C1:
        ch_in1 = c - C0
        ptr = in1_ptr + base_in1 + ch_in1 * HW + hw_offs
        val = tl.load(ptr, mask=hw_mask, other=0.0).to(tl.float32)
        acc = acc + val

    elif c < C0 + C1 + C2:
        ch_in2 = c - C0 - C1
        ptr = in2_ptr + base_in2 + ch_in2 * HW + hw_offs
        val = tl.load(ptr, mask=hw_mask, other=0.0).to(tl.float32)
        acc = acc + val

    else:
        ch_in3 = c - C0 - C1 - C2
        ptr = in3_ptr + base_in3 + ch_in3 * HW + hw_offs
        val = tl.load(ptr, mask=hw_mask, other=0.0).to(tl.float32)
        acc = acc + val

    result = acc.sum(0) / HW
    tl.store(out_ptr + pid, result.to(tl.float16))


# ── Triton kernel: bfloat16 ─────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['C_total'],
)
@triton.jit
def _pool_cat_bf16(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C_total: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C_total
    c = pid % C_total

    hw_offs = tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW

    base_in0 = b * C0 * HW
    base_in1 = b * C1 * HW
    base_in2 = b * C2 * HW
    base_in3 = b * C3 * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    if c < C0:
        ptr = in0_ptr + base_in0 + c * HW + hw_offs
        val = tl.load(ptr, mask=hw_mask, other=0.0).to(tl.float32)
        acc = acc + val

    elif c < C0 + C1:
        ch_in1 = c - C0
        ptr = in1_ptr + base_in1 + ch_in1 * HW + hw_offs
        val = tl.load(ptr, mask=hw_mask, other=0.0).to(tl.float32)
        acc = acc + val

    elif c < C0 + C1 + C2:
        ch_in2 = c - C0 - C1
        ptr = in2_ptr + base_in2 + ch_in2 * HW + hw_offs
        val = tl.load(ptr, mask=hw_mask, other=0.0).to(tl.float32)
        acc = acc + val

    else:
        ch_in3 = c - C0 - C1 - C2
        ptr = in3_ptr + base_in3 + ch_in3 * HW + hw_offs
        val = tl.load(ptr, mask=hw_mask, other=0.0).to(tl.float32)
        acc = acc + val

    result = acc.sum(0) / HW
    tl.store(out_ptr + pid, result.to(tl.bfloat16))


# ── Generic wrapper: dispatches to the correct dtype kernel at runtime ────
@torch.fx.wrap
def pool_cat_flatten_dispatch(in_0, in_1, in_2, in_3):
    B      = in_0.shape[0]
    C0     = in_0.shape[1]   # 320
    C1     = in_1.shape[1]   # 768
    C2     = in_2.shape[1]   # 768
    C3     = in_3.shape[1]   # 192
    C_total = C0 + C1 + C2 + C3  # 2048
    H      = in_0.shape[2]   # 5
    W      = in_0.shape[3]   # 5
    HW     = H * W           # 25
    BLOCK_HW = 32            # next power-of-2 >= 25

    # Output with the same dtype as the inputs (flat [B, C_total])
    out = torch.empty((B, C_total), dtype=in_0.dtype, device=in_0.device)

    grid = (B * C_total,)

    if in_0.dtype == torch.float16:
        _pool_cat_fp16[grid](
            in_0, in_1, in_2, in_3, out,
            C0=C0, C1=C1, C2=C2, C3=C3,
            C_total=C_total, HW=HW, BLOCK_HW=BLOCK_HW,
        )
    else:  # bfloat16
        _pool_cat_bf16[grid](
            in_0, in_1, in_2, in_3, out,
            C0=C0, C1=C1, C2=C2, C3=C3,
            C_total=C_total, HW=HW, BLOCK_HW=BLOCK_HW,
        )

    return out


# ── Pattern / replacement API ────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return pool_cat_flatten_dispatch