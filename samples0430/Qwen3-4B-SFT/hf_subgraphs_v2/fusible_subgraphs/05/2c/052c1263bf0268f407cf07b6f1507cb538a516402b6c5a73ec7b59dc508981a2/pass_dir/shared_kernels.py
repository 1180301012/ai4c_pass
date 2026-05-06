"""
Shared Triton kernels and dispatch wrapper used by ALL AI4C passes.
Both FuseRopeAndExpand_1_1_3_256 and FuseExpandAndReshape_1_1_8_3_256
import `dispatch` from here so that replacement_func() returns the SAME
function object → satisfies output_pass_replacement_func_limit = 1.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: full RoPE + broadcast to 8 heads
#   Grid (8,3): one program per (head, row)
#   Writes tmp_6[n,:] (= rope output) for every head, and out9[h,n,:]
# ---------------------------------------------------------------------------
@triton.jit
def rope_kernel_broadcast(
    in2_ptr,        # key   [1,1,3,256] bf16
    in1_ptr,        # cos   [1,1,3,256] bf16
    in4_ptr,        # sin   [1,1,3,256] bf16
    out6_ptr,       # rope  [1,1,3,256] bf16
    out9_ptr,       # out9  [1,8,3,256] bf16
    D: tl.constexpr,       # 256
    HALF_D: tl.constexpr,  # 128
):
    h_id = tl.program_id(0)   # head 0..7
    n_id = tl.program_id(1)   # row  0..2
    d    = tl.arange(0, D)    # 0..255

    k1   = tl.load(in2_ptr + n_id*D + d).to(tl.float32)
    cos1 = tl.load(in1_ptr + n_id*D + d).to(tl.float32)
    sin1 = tl.load(in4_ptr + n_id*D + d).to(tl.float32)
    k2   = tl.load(in2_ptr + n_id*D + HALF_D + d).to(tl.float32)
    cos2 = tl.load(in1_ptr + n_id*D + HALF_D + d).to(tl.float32)
    sin2 = tl.load(in4_ptr + n_id*D + HALF_D + d).to(tl.float32)

    rope_hi = k1 * cos1 + k2 * sin1
    rope_lo = k1 * cos1 + (-k2) * sin1
    rope    = tl.where(d < HALF_D, rope_lo, rope_hi)

    # Broadcast rope[n,:] to out6[n,:]; same value written 8 times
    for hi in tl.static_range(8):
        tl.store(out6_ptr + n_id*D + d, rope.to(tl.bfloat16))

    # Write out9 for this specific head
    off = h_id * (8 * D) + n_id * D + d
    tl.store(out9_ptr + off, rope.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Kernel B: expand+reshape  [1,1,3,256] → [1,8,3,256]
#   Grid (8,3): one program per (head, row)
#   Copies src[n,:] to dst[h, n, :]
# ---------------------------------------------------------------------------
@triton.jit
def expand_kernel(
    src_ptr,       # tmp_6 or in_5  [1,1,3,256] bf16
    dst_ptr,       # out9            [1,8,3,256] bf16
    D: tl.constexpr,   # 256
    H: tl.constexpr,   # 8
):
    h_id = tl.program_id(0)
    n_id = tl.program_id(1)
    d    = tl.arange(0, D)

    tl.store(dst_ptr + h_id * H * D + n_id * D + d,
             tl.load(src_ptr + n_id * D + d))


# ---------------------------------------------------------------------------
# PER-ROUTE STATE HELPER (dict; set once, then cheap access)
# ---------------------------------------------------------------------------
_r = {}


def _rope(args):
    """args = (key, cos, sin)"""
    out6 = torch.empty_like(args[0])
    out9 = torch.empty(1, 8, 3, 256, dtype=args[0].dtype, device=args[0].device)
    rope_kernel_broadcast[(8, 3)](
        args[0], args[1], args[2],
        out6, out9,
        D=256, HALF_D=128,
        num_warps=4,
    )
    return out6


def _expand(args):
    """args = (x,)  — x is tmp_6 [1,1,3,256] → out9 [1,8,3,256]"""
    out9 = torch.empty(1, 8, 3, 256, dtype=args[0].dtype, device=args[0].device)
    expand_kernel[(8, 3)](
        args[0], out9,
        D=256, H=8,
        num_warps=4,
    )
    return out9


# ---------------------------------------------------------------------------
# SHARED dispatch wrapper — the SAME object returned by EVERY pass file.
# The `route` argument (last arg) is a string appended by replacement_args().
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch(*args):
    route = args[-1]
    if route == "rope":
        return _rope(args[:3])
    elif route == "expand":
        return _expand(args[:1])
    # Fallback (should never happen for valid patterns)
    return args[0]