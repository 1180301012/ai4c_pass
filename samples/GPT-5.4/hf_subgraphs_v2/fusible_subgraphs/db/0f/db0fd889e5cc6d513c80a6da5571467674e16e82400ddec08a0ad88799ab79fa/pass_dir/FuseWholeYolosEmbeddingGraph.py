import torch
import triton
import triton.language as tl


# Pattern matching function
# NOTE: This mirrors model.py exactly and returns every externally observable output.
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_0, in_2, in_1, (2, 2), (0, 0), (1, 1), 1)
    tmp_8 = conv2d.flatten(2)
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = in_3.expand(1, -1, -1)
    tmp_11 = in_4.expand(1, -1, -1)
    tmp_12 = torch.cat((tmp_10, tmp_9, tmp_11), dim=1)
    tmp_13 = in_5[(slice(None, None, None), 0, slice(None, None, None))]
    tmp_14 = tmp_13[(slice(None, None, None), None)]
    tmp_15 = in_5[(slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_16 = in_5[(slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_17 = tmp_16.transpose(1, 2)
    tmp_18 = tmp_17.view(1, 32, 15, 15)
    tmp_19 = torch.nn.functional.interpolate(tmp_18, size=(15, 15), mode='bicubic', align_corners=False)
    tmp_20 = tmp_19.flatten(2)
    tmp_21 = tmp_20.transpose(1, 2)
    tmp_22 = torch.cat((tmp_14, tmp_21, tmp_15), dim=1)
    tmp_23 = tmp_12 + tmp_22
    tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False)
    tmp_25 = in_6[(slice(None, None, None), slice(None, None, None), 0, slice(None, None, None))]
    tmp_26 = tmp_25[(slice(None, None, None), None)]
    tmp_27 = in_6[(slice(None, None, None), slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_28 = in_6[(slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_29 = tmp_28.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    tmp_31 = torch.nn.functional.interpolate(tmp_30, size=(15, 15), mode='bicubic', align_corners=False)
    tmp_32 = tmp_31.flatten(2)
    tmp_33 = tmp_32.transpose(1, 2)
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)
    return (tmp_26, tmp_27, tmp_24, tmp_35)


# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@triton.jit
def _yolos_cls_token_kernel(cls_ptr, pos_ptr, out_ptr):
    c = tl.arange(0, 32)
    cls_v = tl.load(cls_ptr + c).to(tl.float32)
    pos_v = tl.load(pos_ptr + c).to(tl.float32)
    out_v = cls_v + pos_v
    tl.store(out_ptr + c, out_v)


@triton.jit
def _yolos_det_tokens_kernel(det_ptr, pos_ptr, out_ptr):
    pid = tl.program_id(0)
    c = tl.arange(0, 32)
    det_v = tl.load(det_ptr + pid * 32 + c).to(tl.float32)
    pos_v = tl.load(pos_ptr + (226 + pid) * 32 + c).to(tl.float32)
    out_v = det_v + pos_v
    tl.store(out_ptr + (226 + pid) * 32 + c, out_v)


@triton.jit
def _yolos_patch_tokens_kernel(inp_ptr, bias_ptr, weight_ptr, pos_ptr, out_ptr):
    pid = tl.program_id(0)  # 0..224
    c = tl.arange(0, 32)

    oh = pid // 15
    ow = pid % 15
    ih = oh * 2
    iw = ow * 2

    acc = tl.load(bias_ptr + c).to(tl.float32)

    # weight layout: [32, 3, 2, 2], contiguous stride = [12, 4, 2, 1]
    # input layout: [1, 3, 30, 30], contiguous stride = [2700, 900, 30, 1]
    for ic in range(3):
        for kh in range(2):
            for kw in range(2):
                inp_offset = ic * 900 + (ih + kh) * 30 + (iw + kw)
                inp_v = tl.load(inp_ptr + inp_offset).to(tl.float32)
                w_offset = c * 12 + ic * 4 + kh * 2 + kw
                w_v = tl.load(weight_ptr + w_offset).to(tl.float32)
                acc += inp_v * w_v

    pos_v = tl.load(pos_ptr + (1 + pid) * 32 + c).to(tl.float32)
    out_v = acc + pos_v
    tl.store(out_ptr + (1 + pid) * 32 + c, out_v)


@triton.jit
def _copy_mid_position_embeddings_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # output logical shape: [4, 1, 225, 32], contiguous
    # input logical shape:  [4, 1, 236, 32]
    # copy out[b, 0, p, c] = in[b, 0, p + 1, c]
    bc = 225 * 32
    b = offs // bc
    rem = offs % bc
    p = rem // 32
    c = rem % 32

    in_offset = b * (236 * 32) + (p + 1) * 32 + c
    vals = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offs, vals, mask=mask)


@torch.fx.wrap
def fused_whole_yolos_embedding_graph(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # Output tmp_24: [1, 236, 32]
    tmp_24 = torch.empty((1, 236, 32), device=in_0.device, dtype=in_0.dtype)

    # cls token + position embedding -> token 0
    _yolos_cls_token_kernel[(1,)](
        cls_ptr=in_3,
        pos_ptr=in_5,
        out_ptr=tmp_24,
        num_warps=1,
    )

    # patch embeddings conv + position embeddings -> tokens 1..225
    _yolos_patch_tokens_kernel[(225,)](
        inp_ptr=in_0,
        bias_ptr=in_1,
        weight_ptr=in_2,
        pos_ptr=in_5,
        out_ptr=tmp_24,
        num_warps=1,
    )

    # detection tokens + tail position embeddings -> tokens 226..235
    _yolos_det_tokens_kernel[(10,)](
        det_ptr=in_4,
        pos_ptr=in_5,
        out_ptr=tmp_24,
        num_warps=1,
    )

    # tmp_26 and tmp_27 are direct slices/views of in_6.
    tmp_26 = in_6[(slice(None, None, None), slice(None, None, None), slice(0, 1, None), slice(None, None, None))]
    tmp_27 = in_6[(slice(None, None, None), slice(None, None, None), slice(-10, None, None), slice(None, None, None))]

    # tmp_35 is a contiguous materialization of the middle 225 positions.
    tmp_35 = torch.empty((4, 1, 225, 32), device=in_6.device, dtype=in_6.dtype)
    n_elements = 4 * 225 * 32
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _copy_mid_position_embeddings_kernel[grid](
        in_ptr=in_6,
        out_ptr=tmp_35,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )

    return (tmp_26, tmp_27, tmp_24, tmp_35)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_whole_yolos_embedding_graph