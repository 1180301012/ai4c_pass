import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror the source graph exactly.
def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _fused_conv1x1_unfold2x2_stride2_kernel(
    w_ptr,
    x_ptr,
    out_ptr,
    C_OUT: tl.constexpr,
    C_IN: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    PATCHES: tl.constexpr,
    TOTAL_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    mask_m = offs_m < C_OUT
    mask_p = offs_p < PATCHES

    # For H=W=32 and kernel/stride=2, PATCHES = 16 * 16 = 256.
    # Each patch index p maps to top-left spatial coord (2*py, 2*px).
    py = offs_p // (W // 2)
    px = offs_p % (W // 2)
    top_left = py * (2 * W) + px * 2

    acc00 = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)

    for k0 in range(0, C_IN, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < C_IN

        w_ptrs = w_ptr + offs_m[:, None] * C_IN + offs_k[None, :]
        w = tl.load(w_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        x_base = offs_k[:, None] * (H * W) + top_left[None, :]
        valid = mask_k[:, None] & mask_p[None, :]

        x00 = tl.load(x_ptr + x_base, mask=valid, other=0.0)
        x01 = tl.load(x_ptr + x_base + 1, mask=valid, other=0.0)
        x10 = tl.load(x_ptr + x_base + W, mask=valid, other=0.0)
        x11 = tl.load(x_ptr + x_base + W + 1, mask=valid, other=0.0)

        acc00 += tl.dot(w, x00)
        acc01 += tl.dot(w, x01)
        acc10 += tl.dot(w, x10)
        acc11 += tl.dot(w, x11)

    # Output layout is contiguous [1, C_OUT, 4, PATCHES].
    # Flattened per-output-channel segment length is TOTAL_N = 4 * PATCHES.
    out_base = out_ptr + offs_m[:, None] * TOTAL_N
    out_mask = mask_m[:, None] & mask_p[None, :]

    tl.store(out_base + offs_p[None, :], acc00, mask=out_mask)
    tl.store(out_base + (PATCHES + offs_p)[None, :], acc01, mask=out_mask)
    tl.store(out_base + (2 * PATCHES + offs_p)[None, :], acc10, mask=out_mask)
    tl.store(out_base + (3 * PATCHES + offs_p)[None, :], acc11, mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_unfold2x2_stride2_reshape(weight, x):
    # This pass is specialized for the target MobileViTv2 subgraph.
    c_out = weight.shape[0]
    c_in = weight.shape[1]
    n = x.shape[0]
    h = x.shape[2]
    w = x.shape[3]

    # Expected fixed benchmark shape.
    if n != 1:
        raise RuntimeError(f"Expected batch size 1, got {n}")
    if h != 32 or w != 32:
        raise RuntimeError(f"Expected input spatial size 32x32, got {h}x{w}")
    if c_in != 256 or c_out != 128:
        raise RuntimeError(f"Expected weight shape [128, 256, 1, 1], got [{c_out}, {c_in}, 1, 1]")

    patches = (h // 2) * (w // 2)
    total_n = 4 * patches

    out = torch.empty((1, c_out, 4, patches), device=x.device, dtype=x.dtype)

    # Tuned for this small fixed problem size.
    BLOCK_M = 32
    BLOCK_P = 32
    BLOCK_K = 32

    grid = (triton.cdiv(c_out, BLOCK_M), triton.cdiv(patches, BLOCK_P))

    _fused_conv1x1_unfold2x2_stride2_kernel[grid](
        weight,
        x,
        out,
        C_OUT=c_out,
        C_IN=c_in,
        H=h,
        W=w,
        PATCHES=patches,
        TOTAL_N=total_n,
        BLOCK_M=BLOCK_M,
        BLOCK_P=BLOCK_P,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv1x1_unfold2x2_stride2_reshape