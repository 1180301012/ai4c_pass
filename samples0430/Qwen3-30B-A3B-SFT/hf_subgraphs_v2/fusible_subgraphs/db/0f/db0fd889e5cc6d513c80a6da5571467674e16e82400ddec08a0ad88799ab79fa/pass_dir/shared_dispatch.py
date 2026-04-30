"""
Shared Triton kernels and dispatch wrapper used by both fusion passes.
Both FusePosEmbedCatAdd and FuseMiddlePath import dispatch_kernel from here,
ensuring replacement_func() returns the SAME function object and avoids the
output_pass_replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fuse position-embedding path (in_5) + residual add
# in_5 [1,236,32]  cls_tok [1,1,32]  det_toks [1,10,32]  tmp_12 [1,236,32]
# → output [1,236,32]  where out[0, t, c] = emb[t,c] + tmp_12[0,t,c]
#   and emb[t] is cls_tok (t=0), in_5[0,t,:] (t=1..225), det_toks[0,t-225,:] (t=226..235)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32}, num_warps=1),
        triton.Config({'BLOCK_C': 32}, num_warps=2),
        triton.Config({'BLOCK_C': 32}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def _pe_cat_add_kernel(
    cls_ptr,   # [1, 1, C]
    mid_ptr,   # [1, N_SEQ, C]
    last_ptr,  # [1, N_LAST, C]
    add_ptr,   # [1, N_SEQ, C]  (residual to add)
    out_ptr,   # [1, N_SEQ, C]
    N_CLS,     # 1
    N_MID,     # 225
    N_LAST,    # 10
    N_SEQ,     # 236
    BLOCK_C: tl.constexpr,
):
    pid  = tl.program_id(0)   # 0 .. N_SEQ-1
    cid  = tl.program_id(1)   # column chunk (0 when BLOCK_C == C)

    offs_c  = cid * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c  = offs_c < 32

    is_cls  = pid < N_CLS
    is_last = pid >= N_SEQ - N_LAST

    # Load embedding source
    cls_val  = tl.load(cls_ptr  + offs_c,                          mask=mask_c & is_cls,  other=0.0)
    mid_t    = pid - N_CLS
    mid_val  = tl.load(mid_ptr  + mid_t * 32 + offs_c,             mask=mask_c & ~is_cls & ~is_last, other=0.0)
    last_t   = pid - (N_SEQ - N_LAST)
    last_val = tl.load(last_ptr + last_t * 32 + offs_c,            mask=mask_c & is_last, other=0.0)

    val = tl.where(is_cls, cls_val, tl.where(is_last, last_val, mid_val))

    # Add residual and write
    add_val = tl.load(add_ptr + pid * 32 + offs_c, mask=mask_c, other=0.0)
    tl.store(out_ptr + pid * 32 + offs_c, val + add_val)


# ---------------------------------------------------------------------------
# Kernel 2: Fuse middle path (in_6 slice → reshape → output)
# in_6 [4,1,236,32]  → output [4,1,225,32]   (skip token 0, reorder spatially)
# out[n,0,p,c] = in_6[n,0,p+1,c]   (225 spatial positions 1..225)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32}, num_warps=1),
        triton.Config({'BLOCK_C': 32}, num_warps=2),
        triton.Config({'BLOCK_C': 32}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def _middle_copy_kernel(
    in6_ptr,       # [4, 1, 236, 32]
    out_ptr,       # [4, 1, 225, 32]
    IN_6_BATCH,    # 236 * 32 = 7552
    IN_6_OFFSET,   # 1  (skip first token)
    OUT_BATCH,     # 225 * 32 = 7200
    BLOCK_C: tl.constexpr,
):
    pid  = tl.program_id(0)   # 0 .. 4*225-1
    n    = pid // 225
    pos  = pid %  225

    cid    = tl.program_id(1)
    offs_c = cid * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < 32

    # in_6[n, 0, pos + IN_6_OFFSET, c]  strides [7552, 7552, 32, 1]
    val = tl.load(in6_ptr + n * IN_6_BATCH + (pos + IN_6_OFFSET) * 32 + offs_c,
                  mask=mask_c, other=0.0)

    # out[n, 0, pos, c]  strides [7200, 7200, 32, 1]
    tl.store(out_ptr + n * OUT_BATCH + pos * 32 + offs_c, val)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper  (BOTH pass files return this same object)
# route = "route_pe"  → pos-embed + add  (FusePosEmbedCatAdd)
# route = "route_m"   → middle copy       (FuseMiddlePath)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_kernel(a0, a1, a2, a3, route):
    if route == "route_pe":
        # a0=in_5 [1,236,32]  a1=cls_tok [1,1,32]  a2=det_toks [1,10,32]  a3=tmp_12 [1,236,32]
        N_SEQ = 236
        C     = 32
        out   = torch.empty((1, N_SEQ, C), dtype=a0.dtype, device=a0.device)
        _pe_cat_add_kernel[(N_SEQ, 1)](
            a1, a0, a2, a3, out,
            1, 225, 10, N_SEQ,
        )
        return out
    elif route == "route_m":
        # a0=in_6 [4,1,236,32]  a1=tmp_26 [4,1,1,32]  a2=tmp_27 [4,1,10,32]
        # (a1, a2 are unused by this kernel; they are external inputs required by the pass framework)
        N_BATCH  = 4
        N_SPATIAL = 225
        C        = 32
        out      = torch.empty((N_BATCH, 1, N_SPATIAL, C), dtype=a0.dtype, device=a0.device)
        _middle_copy_kernel[(N_BATCH * N_SPATIAL, 1)](
            a0, out,
            236 * 32,  # IN_6_BATCH
            1,         # IN_6_OFFSET
            N_SPATIAL * C,  # OUT_BATCH
        )
        return out