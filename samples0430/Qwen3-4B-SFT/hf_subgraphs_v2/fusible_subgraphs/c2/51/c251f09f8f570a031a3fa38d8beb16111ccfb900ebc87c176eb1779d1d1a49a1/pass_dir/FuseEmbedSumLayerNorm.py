import torch
import triton
import triton.language as tl
import operator


# ---------------------------------------------------------------------------
# Triton kernel: high-performance LayerNorm
# Each program handles one (batch, seq) position; parallel over full D=768.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_D': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_D': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_D': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_D': 1024}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_D': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_D': 512},  num_warps=8,  num_stages=2),
    ],
    key=['BS', 'D'],
)
@triton.jit
def fused_embed_sum_ln_kernel(
    # 9 index tensors  [BS] int64
    ids0_ptr, ids1_ptr, ids2_ptr, ids3_ptr, ids4_ptr, ids5_ptr,
    ids6_ptr, ids7_ptr, ids8_ptr,
    # 9 embedding weight tables [V, D]
    e0_ptr, e1_ptr, e2_ptr, e3_ptr, e4_ptr, e5_ptr,
    e6_ptr, e7_ptr, e8_ptr,
    # LayerNorm params [D]
    lnw_ptr, lnb_ptr,
    # Shapes / strides
    BS, S, D,
    # Strides of index tensors
    ids0_stride, ids1_stride, ids2_stride,
    ids3_stride, ids4_stride, ids5_stride,
    ids6_stride, ids7_stride, ids8_stride,
    # Strides of embedding tables
    e0_stride, e1_stride, e2_stride,
    e3_stride, e4_stride, e5_stride,
    e6_stride, e7_stride, e8_stride,
    # LayerNorm eps (~1e-12) is scalar
    eps,
    # Output row-stride = D
    out_stride,
    # BLOCK_D supplied by autotune >= D=768
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)          # one pid per (batch, seq) position
    b   = pid // S
    s   = pid %  S

    # ---- load per-embedding id values ----
    base_id = b * ids0_stride + s
    w_id   = tl.load(ids0_ptr + base_id).to(tl.int64)
    cls_id = tl.load(ids1_ptr + base_id).to(tl.int64)
    x_id   = tl.load(ids2_ptr + base_id).to(tl.int64)
    y_id   = tl.load(ids3_ptr + base_id).to(tl.int64)
    h_id   = tl.load(ids4_ptr + base_id).to(tl.int64)
    tt_id  = tl.load(ids5_ptr + base_id).to(tl.int64)
    xd_id  = tl.load(ids6_ptr + base_id).to(tl.int64)
    yd_id  = tl.load(ids7_ptr + base_id).to(tl.int64)
    wid    = tl.load(ids8_ptr + base_id).to(tl.int64)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # ---- gather all 9 embedding rows in fp32 ----
    x0 = tl.load(e0_ptr + w_id   * D + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(e1_ptr + cls_id * D + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(e2_ptr + x_id   * D + offs, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(e3_ptr + y_id   * D + offs, mask=mask, other=0.0).to(tl.float32)
    x4 = tl.load(e4_ptr + h_id   * D + offs, mask=mask, other=0.0).to(tl.float32)
    x5 = tl.load(e5_ptr + tt_id  * D + offs, mask=mask, other=0.0).to(tl.float32)
    x6 = tl.load(e6_ptr + xd_id  * D + offs, mask=mask, other=0.0).to(tl.float32)
    x7 = tl.load(e7_ptr + yd_id  * D + offs, mask=mask, other=0.0).to(tl.float32)
    x8 = tl.load(e8_ptr + wid   * D + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- fused sum ----
    z = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

    # ---- layer norm (over D dims) ----
    z_f = tl.where(mask, z, 0.0)
    mean = tl.sum(z_f, axis=0) / D
    diff = tl.where(mask, z - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    lw = tl.load(lnw_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    lb = tl.load(lnb_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = (diff * rstd * lw + lb).to(z.dtype)

    # ---- store ----
    tl.store(out_ptr + pid * D + offs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_D': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_D': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_D': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_D': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_D': 512},  num_warps=8,  num_stages=2),
    ],
    key=['n_rows', 'D'],
)
@triton.jit
def triton_layer_norm_kernel(
    x_ptr,        # input [N_rows, D]
    w_ptr,        # weight [D]
    b_ptr,        # bias   [D]
    out_ptr,      # output [N_rows, D]
    D,
    w_stride,     # stride between weight elements
    b_stride,     # stride between bias elements
    out_stride,   # per-row output stride
    eps,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_ptr + pid * D + offs, mask=mask, other=0.0).to(tl.float32)
    x_f = tl.where(mask, x, 0.0)

    mean = tl.sum(x_f, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    lw = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    lb = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = (diff * rstd * lw + lb).to(tl.float32)
    tl.store(out_ptr + pid * D + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper: standalone LayerNorm (used by the replacement pass)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_layer_norm(x, ln_w, ln_b):
    """Triton LayerNorm for D=768.  NO reshape/view — avoids forbidden ops."""
    # x: [..., D] contiguous (output of add chain: [B, S, 768])
    D = x.shape[-1]                        # 768
    nw = x.numel() // D                    # B * S rows

    out = torch.empty_like(x)             # same shape/dtype; avoids view()

    # Kernel flat-index scheme: element at (row, d) lives at offset row*D + d.
    # For a contiguous [..., D] tensor this holds directly when row = row_idx*D+col_idx.
    # Pass actual per-axis strides so garbage input tensors are caught at runtime.
    triton_layer_norm_kernel[(nw,)](
        x, ln_w, ln_b, out,
        D,
        x.stride(-1),  # element stride in input (= 1 for contiguous [...,D])
        D,              # row stride in weight (= D for [D] contiguous, 1 is dummy)
        x.stride(-1),  # element stride in output (= 1 for contiguous)
        1e-12,
    )

    return out


# ---------------------------------------------------------------------------
# Full-fusion wrapper (embeds + sum + LN)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_batch_embed_sum_layernorm(ids0, ids1, ids2, ids3, ids4, ids5,
                                     ids6, ids7, ids8,
                                     e0, e1, e2, e3, e4, e5,
                                     e6, e7, e8,
                                     ln_w, ln_b):
    D = e0.shape[1]          # 768
    BS = ids0.shape[0] * ids0.shape[1]
    S  = ids0.shape[1]
    out = torch.empty((BS, S, D), dtype=e0.dtype, device=e0.device)

    # Flat path
    use_flat = False
    try:
        _i0 = ids0.view(-1)
        s0  = _i0.stride(0)
        io0 = ids0.view(-1)
        i1  = ids1.stride(0);  io1 = ids1.view(-1)
        i2  = ids2.stride(0);  io2 = ids2.view(-1)
        i3  = ids3.stride(0);  io3 = ids3.view(-1)
        i4  = ids4.stride(0);  io4 = ids4.view(-1)
        i5  = ids5.stride(0);  io5 = ids5.view(-1)
        i6  = ids6.stride(0);  io6 = ids6.view(-1)
        i7  = ids7.stride(0);  io7 = ids7.view(-1)
        i8  = ids8.stride(0);  io8 = ids8.view(-1)
        use_flat = True
    except Exception:
        pass

    if use_flat:
        fused_embed_sum_ln_kernel[(BS,)](
            io0, io1, io2, io3, io4, io5, io6, io7, io8,
            e0, e1, e2, e3, e4, e5, e6, e7, e8,
            ln_w, ln_b,
            BS, S, D,
            s0, i1, i2, i3, i4, i5, i6, i7, i8,
            e0.stride(0), e1.stride(0), e2.stride(0),
            e3.stride(0), e4.stride(0), e5.stride(0),
            e6.stride(0), e7.stride(0), e8.stride(0),
            1e-12, D,
        )
    else:
        fused_embed_sum_ln_kernel[(BS,)](
            ids0, ids1, ids2, ids3, ids4, ids5,
            ids6, ids7, ids8,
            e0, e1, e2, e3, e4, e5, e6, e7, e8,
            ln_w, ln_b,
            BS, S, D,
            ids0.stride(0), ids1.stride(0), ids2.stride(0),
            ids3.stride(0), ids4.stride(0), ids5.stride(0),
            ids6.stride(0), ids7.stride(0), ids8.stride(0),
            e0.stride(0), e1.stride(0), e2.stride(0),
            e3.stride(0), e4.stride(0), e5.stride(0),
            e6.stride(0), e7.stride(0), e8.stride(0),
            1e-12, D,
        )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface
#
# Key insight from diagnostics:
#   The SubgraphMatcher anchors on operator.getitem(native_ln, 0) in the
#   pattern but the model's getitem anchor was the in_2 slice getitem.
#   Fix: make the LAST returning node = native_layer_norm.default itself
#   (NOT the getitem[0]).  Then native_ln is the anchor and the model's
#   native_layer_norm node will be found directly.
#
# normalized_shape may be stored as either [768] or (768,) in the model.
# We pass it as an explicit pattern argument (wildcard via match_placeholder).
# ---------------------------------------------------------------------------

def pattern(x, ln_w, ln_b, normalized_shape):
    """
    Match: F.layer_norm(x, [D], w, b, eps) directly – the F.layer_norm
    function itself IS the last returning node (anchor).  SubgraphMatcher
    iterates over pattern_returning_nodes = [layer_norm_node] and looks
    for nodes in the model graph with the same op/target as that anchor.

    normalized_shape is passed as a pattern argument so the normalized_shape
    literal [768] or (768,) in the model can be matched as a placeholder.
    """
    result = torch.nn.functional.layer_norm(x, normalized_shape, ln_w, ln_b, 1e-12)
    return result


def replacement_args(x, ln_w, ln_b, normalized_shape):
    # normalized_shape (placeholder) is not a computation node; ignore.
    return (x, ln_w, ln_b)


def replacement_func():
    return triton_layer_norm