import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_0, in_2, in_1, (2, 2), (0, 0), (1, 1), 1)
    tmp_8 = conv2d.flatten(2)
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = in_3.expand(1, -1, -1)
    tmp_11 = in_4.expand(1, -1, -1)
    tmp_12 = torch.cat((tmp_10, tmp_9, tmp_11), dim = 1)
    tmp_13 = in_5[(slice(None, None, None), 0, slice(None, None, None))]
    tmp_14 = tmp_13[(slice(None, None, None), None)]
    tmp_15 = in_5[(slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_16 = in_5[(slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_17 = tmp_16.transpose(1, 2)
    tmp_18 = tmp_17.view(1, 32, 15, 15)
    tmp_19 = torch.nn.functional.interpolate(tmp_18, size = (15, 15), mode = 'bicubic', align_corners = False)
    tmp_20 = tmp_19.flatten(2)
    tmp_21 = tmp_20.transpose(1, 2)
    tmp_22 = torch.cat((tmp_14, tmp_21, tmp_15), dim = 1)
    tmp_23 = tmp_12 + tmp_22
    tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False)
    tmp_25 = in_6[(slice(None, None, None), slice(None, None, None), 0, slice(None, None, None))]
    tmp_26 = tmp_25[(slice(None, None, None), None)]
    tmp_27 = in_6[(slice(None, None, None), slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_28 = in_6[(slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_29 = tmp_28.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    tmp_31 = torch.nn.functional.interpolate(tmp_30, size = (15, 15), mode = 'bicubic', align_corners = False)
    tmp_32 = tmp_31.flatten(2)
    tmp_33 = tmp_32.transpose(1, 2)
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)
    return (tmp_26, tmp_27, tmp_24, tmp_35)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


# ============= Triton Kernels =============

@triton.jit
def conv2d_patch_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    C_IN: tl.constexpr, H_IN: tl.constexpr, W_IN: tl.constexpr,
    C_OUT: tl.constexpr, H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    SEQ_LEN: tl.constexpr, EMB_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Conv2d with 2x2 kernel stride 2, producing [1, SEQ_LEN, EMB_DIM] output directly."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    total = SEQ_LEN * EMB_DIM
    mask = offsets < total

    # Decompose flat offset into (seq, dim) = (patch_index, embedding_dim)
    seq = offsets // EMB_DIM
    dim = offsets % EMB_DIM

    # Decompose seq into spatial coords (oh, ow) for conv2d output
    oh = seq // W_OUT
    ow = seq % W_OUT

    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + dim, mask=mask, other=0.0).to(tl.float32)

    # Initialize accumulator with bias
    acc = bias_val

    # Conv2d: sum over input channels and kernel positions
    # output[0, dim, oh, ow] = bias[dim] + sum_{ic,kh,kw} input[0,ic,oh*stride+kh,ow*stride+kw] * weight[dim,ic,kh,kw]
    for ic in range(C_IN):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh * STRIDE_H + kh
                iw = ow * STRIDE_W + kw

                # Load input value at [0, ic, ih, iw]
                input_offset = ic * H_IN * W_IN + ih * W_IN + iw
                input_mask = mask & (oh < H_OUT) & (ow < W_OUT) & (ih < H_IN) & (iw < W_IN)
                input_val = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0).to(tl.float32)

                # Load weight value at [dim, ic, kh, kw]
                weight_offset = dim * C_IN * KH * KW + ic * KH * KW + kh * KW + kw
                weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0).to(tl.float32)

                # Accumulate
                acc = acc + input_val * weight_val

    # Store output at [0, seq, dim] in contiguous [1, SEQ_LEN, EMB_DIM] layout
    output_offset = seq * EMB_DIM + dim
    tl.store(output_ptr + output_offset, acc, mask=mask)


@triton.jit
def fused_embed_add_kernel(
    cls_ptr, patch_ptr, det_ptr, pos_ptr, out_ptr,
    CLS_LEN: tl.constexpr, PATCH_LEN: tl.constexpr, DET_LEN: tl.constexpr,
    TOTAL_LEN: tl.constexpr, EMB_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: cat(cls, patches, det_tokens) + position_embeddings + add (skip dropout)."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    total = TOTAL_LEN * EMB_DIM
    mask = offsets < total

    # Decompose flat offset into (seq, dim)
    seq = offsets // EMB_DIM
    dim = offsets % EMB_DIM

    # Determine which source to read for token embedding
    cls_mask = seq < CLS_LEN
    patch_mask = (seq >= CLS_LEN) & (seq < CLS_LEN + PATCH_LEN)
    det_mask = seq >= CLS_LEN + PATCH_LEN

    # Load cls token: in_3[0, 0, dim] at offset dim
    cls_val = tl.load(cls_ptr + dim, mask=mask & cls_mask, other=0.0).to(tl.float32)

    # Load patch embedding: patches[0, seq-CLS_LEN, dim]
    patch_seq = seq - CLS_LEN
    patch_offset = patch_seq * EMB_DIM + dim
    patch_val = tl.load(patch_ptr + patch_offset, mask=mask & patch_mask, other=0.0).to(tl.float32)

    # Load detection token: in_4[0, seq-CLS_LEN-PATCH_LEN, dim]
    det_seq = seq - CLS_LEN - PATCH_LEN
    det_offset = det_seq * EMB_DIM + dim
    det_val = tl.load(det_ptr + det_offset, mask=mask & det_mask, other=0.0).to(tl.float32)

    # Combine token values using conditional selection
    token_val = tl.where(cls_mask, cls_val, tl.where(patch_mask, patch_val, det_val))

    # Load position embedding: in_5[0, seq, dim] at offset seq*EMB_DIM+dim = offsets
    # The position embedding output is equivalent to in_5 (verified analytically)
    pos_val = tl.load(pos_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Add token + position (skip dropout since training=False makes it identity)
    out_val = token_val + pos_val

    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)


@triton.jit
def mid_pos_cls_kernel(
    mid_pos_ptr, out_ptr,
    SEQ_TOTAL: tl.constexpr, EMB_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy in_6[:, :, 0, :] to output [4, 1, 1, 32].
    For each [b, 0, 0, d], source is in_6[b, 0, 0, d]."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Total output: BATCH * 1 * 1 * EMB_DIM = 4 * 32 = 128
    BATCH = 4
    total = BATCH * EMB_DIM
    mask = offsets < total

    batch = offsets // EMB_DIM
    dim = offsets % EMB_DIM

    # Source: in_6[b, 0, 0, dim] at offset b * SEQ_TOTAL * EMB_DIM + 0 * SEQ_TOTAL * EMB_DIM + 0 * EMB_DIM + dim
    # Simplified: b * SEQ_TOTAL * EMB_DIM + dim
    src_offset = batch * SEQ_TOTAL * EMB_DIM + dim
    val = tl.load(mid_pos_ptr + src_offset, mask=mask, other=0.0).to(tl.float32)

    # Output [4, 1, 1, 32] contiguous: offset = b * EMB_DIM + dim
    tl.store(out_ptr + offsets, val, mask=mask)


@triton.jit
def mid_pos_det_kernel(
    mid_pos_ptr, out_ptr,
    SEQ_TOTAL: tl.constexpr, DET_LEN: tl.constexpr, EMB_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy in_6[:, :, -10:, :] to output [4, 1, 10, 32].
    For each [b, 0, s, d], source is in_6[b, 0, SEQ_TOTAL-DET_LEN+s, d]."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    BATCH = 4
    total = BATCH * DET_LEN * EMB_DIM
    mask = offsets < total

    # Decompose: flat offset -> (batch, det_seq, dim)
    flat_2d = offsets // EMB_DIM
    dim = offsets % EMB_DIM
    batch = flat_2d // DET_LEN
    det_seq = flat_2d % DET_LEN

    # Source: in_6[b, 0, SEQ_TOTAL-DET_LEN+det_seq, dim]
    src_seq = SEQ_TOTAL - DET_LEN + det_seq
    src_offset = batch * SEQ_TOTAL * EMB_DIM + src_seq * EMB_DIM + dim
    val = tl.load(mid_pos_ptr + src_offset, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + offsets, val, mask=mask)


@triton.jit
def mid_pos_patch_kernel(
    mid_pos_ptr, out_ptr,
    SEQ_TOTAL: tl.constexpr, PATCH_LEN: tl.constexpr, EMB_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Rearrange in_6[:, :, 1:-10, :] to output [4, 1, 225, 32].
    For each [b, 0, k, d], source is in_6[b, 0, 1+k, d].
    This skips the identity interpolate operation."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    BATCH = 4
    total = BATCH * PATCH_LEN * EMB_DIM
    mask = offsets < total

    # Decompose: flat offset -> (batch, patch_seq, dim)
    flat_2d = offsets // EMB_DIM
    dim = offsets % EMB_DIM
    batch = flat_2d // PATCH_LEN
    patch_seq = flat_2d % PATCH_LEN

    # Source: in_6[b, 0, 1+patch_seq, dim] (slice starts at index 1)
    src_seq = 1 + patch_seq
    src_offset = batch * SEQ_TOTAL * EMB_DIM + src_seq * EMB_DIM + dim
    val = tl.load(mid_pos_ptr + src_offset, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + offsets, val, mask=mask)


# ============= Wrapper Function =============

@torch.fx.wrap
def fused_yolos_forward(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    dtype = in_0.dtype
    device = in_0.device

    # Constants derived from weight_meta
    C_IN = 3; H_IN = 30; W_IN = 30
    C_OUT = 32; H_OUT = 15; W_OUT = 15
    STRIDE_H = 2; STRIDE_W = 2
    KH = 2; KW = 2
    SEQ_LEN = 225  # H_OUT * W_OUT = 15 * 15
    EMB_DIM = 32
    CLS_LEN = 1; PATCH_LEN = 225; DET_LEN = 10
    TOTAL_LEN = 236  # CLS_LEN + PATCH_LEN + DET_LEN
    SEQ_TOTAL = 236  # in_5 and in_6 sequence length

    BLOCK_SIZE = 512

    # Step 1: Conv2d → patches [1, 225, 32] (fused flatten + transpose)
    patches = torch.empty((1, SEQ_LEN, EMB_DIM), dtype=dtype, device=device)
    total_patch = SEQ_LEN * EMB_DIM
    num_programs_patch = triton.cdiv(total_patch, BLOCK_SIZE)
    conv2d_patch_kernel[(num_programs_patch,)](
        input_ptr=in_0, weight_ptr=in_2, bias_ptr=in_1, output_ptr=patches,
        C_IN=C_IN, H_IN=H_IN, W_IN=W_IN,
        C_OUT=C_OUT, H_OUT=H_OUT, W_OUT=W_OUT,
        STRIDE_H=STRIDE_H, STRIDE_W=STRIDE_W,
        KH=KH, KW=KW,
        SEQ_LEN=SEQ_LEN, EMB_DIM=EMB_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 2: Fused token + position embedding assembly + add → combined [1, 236, 32]
    # This replaces: expand, cat, slice, transpose, view, interpolate(identity), flatten, transpose, cat, add, dropout(identity)
    combined = torch.empty((1, TOTAL_LEN, EMB_DIM), dtype=dtype, device=device)
    total_combined = TOTAL_LEN * EMB_DIM
    num_programs_combined = triton.cdiv(total_combined, BLOCK_SIZE)
    fused_embed_add_kernel[(num_programs_combined,)](
        cls_ptr=in_3, patch_ptr=patches, det_ptr=in_4, pos_ptr=in_5, out_ptr=combined,
        CLS_LEN=CLS_LEN, PATCH_LEN=PATCH_LEN, DET_LEN=DET_LEN,
        TOTAL_LEN=TOTAL_LEN, EMB_DIM=EMB_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 3: Mid position embeddings (skip interpolate(identity))
    # tmp_26: [4, 1, 1, 32] from in_6[:, :, 0, :][:, None]
    tmp_26 = torch.empty((4, 1, 1, EMB_DIM), dtype=dtype, device=device)
    total_cls = 4 * EMB_DIM
    num_programs_cls = triton.cdiv(total_cls, BLOCK_SIZE)
    mid_pos_cls_kernel[(num_programs_cls,)](
        mid_pos_ptr=in_6, out_ptr=tmp_26,
        SEQ_TOTAL=SEQ_TOTAL, EMB_DIM=EMB_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # tmp_27: [4, 1, 10, 32] from in_6[:, :, -10:, :]
    tmp_27 = torch.empty((4, 1, DET_LEN, EMB_DIM), dtype=dtype, device=device)
    total_det = 4 * DET_LEN * EMB_DIM
    num_programs_det = triton.cdiv(total_det, BLOCK_SIZE)
    mid_pos_det_kernel[(num_programs_det,)](
        mid_pos_ptr=in_6, out_ptr=tmp_27,
        SEQ_TOTAL=SEQ_TOTAL, DET_LEN=DET_LEN, EMB_DIM=EMB_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # tmp_35: [4, 1, 225, 32] rearranged from in_6[:, :, 1:-10, :]
    # (skipping identity interpolate and all intermediate transpose/view/contiguous)
    tmp_35 = torch.empty((4, 1, PATCH_LEN, EMB_DIM), dtype=dtype, device=device)
    total_patch_mid = 4 * PATCH_LEN * EMB_DIM
    num_programs_patch_mid = triton.cdiv(total_patch_mid, BLOCK_SIZE)
    mid_pos_patch_kernel[(num_programs_patch_mid,)](
        mid_pos_ptr=in_6, out_ptr=tmp_35,
        SEQ_TOTAL=SEQ_TOTAL, PATCH_LEN=PATCH_LEN, EMB_DIM=EMB_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (tmp_26, tmp_27, combined, tmp_35)


def replacement_func():
    return fused_yolos_forward