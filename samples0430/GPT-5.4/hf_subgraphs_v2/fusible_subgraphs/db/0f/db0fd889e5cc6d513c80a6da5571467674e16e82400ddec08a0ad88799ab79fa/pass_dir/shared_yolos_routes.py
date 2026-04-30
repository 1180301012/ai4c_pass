import torch
import triton
import triton.language as tl


@triton.jit
def yolos_cat_add_kernel(
    cls_ptr,
    patch_ptr,
    det_ptr,
    pos_ptr,
    out_ptr,
    cls_s0,
    cls_s1,
    cls_s2,
    patch_s0,
    patch_s1,
    patch_s2,
    det_s0,
    det_s1,
    det_s2,
    pos_s0,
    pos_s1,
    pos_s2,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, BLOCK_C)
    mask = (offs_t[:, None] < 236) & (offs_c[None, :] < 32)

    pos = tl.load(
        pos_ptr + offs_t[:, None] * pos_s1 + offs_c[None, :] * pos_s2,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    acc = pos

    cls_mask = offs_t == 0
    det_mask = offs_t >= 226
    patch_mask = (offs_t >= 1) & (offs_t < 226)

    cls = tl.load(
        cls_ptr + offs_c * cls_s2,
        mask=offs_c < 32,
        other=0.0,
    ).to(tl.float32)
    acc += tl.where(cls_mask[:, None], cls[None, :], 0.0)

    patch_idx = offs_t - 1
    patch = tl.load(
        patch_ptr + patch_idx[:, None] * patch_s1 + offs_c[None, :] * patch_s2,
        mask=patch_mask[:, None] & (offs_c[None, :] < 32),
        other=0.0,
    ).to(tl.float32)
    acc += patch

    det_idx = offs_t - 226
    det = tl.load(
        det_ptr + det_idx[:, None] * det_s1 + offs_c[None, :] * det_s2,
        mask=det_mask[:, None] & (offs_c[None, :] < 32),
        other=0.0,
    ).to(tl.float32)
    acc += det

    tl.store(
        out_ptr + offs_t[:, None] * 32 + offs_c[None, :],
        acc,
        mask=mask,
    )


@triton.jit
def yolos_copy_225x32_kernel(
    src_ptr,
    out_ptr,
    src_batch_stride,
    src_token_stride,
    src_chan_stride,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, BLOCK_C)
    mask = (offs_t[:, None] < 225) & (offs_c[None, :] < 32)

    src_offsets = pid_b * src_batch_stride + offs_t[:, None] * src_token_stride + offs_c[None, :] * src_chan_stride
    out_offsets = pid_b * 225 * 32 + offs_t[:, None] * 32 + offs_c[None, :]

    x = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, x, mask=mask)


@torch.fx.wrap
def yolos_dispatch(*args):
    route = args[-1]

    if route == "cat_add" or route == "cat_add_dropout":
        cls_tokens, patch_tokens, det_tokens, pos_tokens = args[:-1]
        out = torch.empty((1, 236, 32), device=patch_tokens.device, dtype=patch_tokens.dtype)
        BLOCK_T = 32
        BLOCK_C = 32
        grid = (triton.cdiv(236, BLOCK_T),)
        yolos_cat_add_kernel[grid](
            cls_tokens,
            patch_tokens,
            det_tokens,
            pos_tokens,
            out,
            cls_tokens.stride(0),
            cls_tokens.stride(1),
            cls_tokens.stride(2),
            patch_tokens.stride(0),
            patch_tokens.stride(1),
            patch_tokens.stride(2),
            det_tokens.stride(0),
            det_tokens.stride(1),
            det_tokens.stride(2),
            pos_tokens.stride(0),
            pos_tokens.stride(1),
            pos_tokens.stride(2),
            BLOCK_T=BLOCK_T,
            BLOCK_C=BLOCK_C,
        )
        return out

    if route == "copy225_contiguous_view" or route == "copy225_clone_view":
        src = args[0]
        out = torch.empty((4, 1, 225, 32), device=src.device, dtype=src.dtype)
        if src.dim() == 4:
            batch_stride = src.stride(0)
            token_stride = src.stride(2)
            chan_stride = src.stride(3)
        else:
            batch_stride = src.stride(0)
            token_stride = src.stride(1)
            chan_stride = src.stride(2)
        BLOCK_T = 32
        BLOCK_C = 32
        grid = (triton.cdiv(225, BLOCK_T), 4)
        yolos_copy_225x32_kernel[grid](
            src,
            out,
            batch_stride,
            token_stride,
            chan_stride,
            BLOCK_T=BLOCK_T,
            BLOCK_C=BLOCK_C,
        )
        return out

    raise ValueError(f"Unknown route: {route}")