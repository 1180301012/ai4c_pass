import torch
import triton
import triton.language as tl


@triton.jit
def _swin_roll_copy_kernel(
    in3_ptr,
    in3_s0, in3_s1, in3_s2, in3_s3, in3_s4, in3_s5,
    out_ptr,
    out_s0, out_s1, out_s2,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_cb = tl.program_id(1)

    h = pid_row // W
    w = pid_row - h * W

    src_h = (h + H - SHIFT) % H
    src_w = (w + W - SHIFT) % W

    h_outer = src_h // 8
    h_inner = src_h % 8
    w_outer = src_w // 8
    w_inner = src_w % 8

    offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_c < C

    src_ptrs = (
        in3_ptr
        + h_outer * in3_s1
        + h_inner * in3_s2
        + w_outer * in3_s3
        + w_inner * in3_s4
        + offs_c * in3_s5
    )
    x = tl.load(src_ptrs, mask=mask)

    dst_ptrs = out_ptr + pid_row * out_s1 + offs_c * out_s2
    tl.store(dst_ptrs, x, mask=mask)


def _launch_swin_roll_copy(in3, h, w, c):
    out = torch.empty((1, h * w, c), device=in3.device, dtype=in3.dtype)

    in3_s = in3.stride()
    out_s = out.stride()

    block_c = 256 if c >= 256 else 128
    grid = (h * w, triton.cdiv(c, block_c))

    _swin_roll_copy_kernel[grid](
        in3,
        in3_s[0], in3_s[1], in3_s[2], in3_s[3], in3_s[4], in3_s[5],
        out,
        out_s[0], out_s[1], out_s[2],
        H=h,
        W=w,
        C=c,
        SHIFT=4,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )
    return out


@torch.fx.wrap
def shared_swin_roll_layernorm_add(in3, route):
    if route == "swin_roll_prefix_64_64_384":
        return _launch_swin_roll_copy(in3, 64, 64, 384)
    if route == "swin_roll_prefix_32_32_768":
        return _launch_swin_roll_copy(in3, 32, 32, 768)
    raise RuntimeError(f"Unknown route: {route}")