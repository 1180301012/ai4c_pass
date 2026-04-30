import torch
import triton
import triton.language as tl


# ============ Conv1x1 Kernel ============
@triton.jit
def conv1x1_kernel(
    feat_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C_in, C_out, HW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(HW, BLOCK_N)
    batch_id = pid // num_n
    n_id = pid % num_n

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < C_out
    mask_n = offs_n < HW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    bias_val = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0)
    acc += bias_val[:, None]

    feat_base = feat_ptr + batch_id * C_in * HW
    for k in range(0, C_in, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < C_in
        w = tl.load(weight_ptr + offs_m[:, None] * C_in + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        inp = tl.load(feat_base + offs_k[:, None] * HW + offs_n[None, :],
                      mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(w, inp)

    out_base = out_ptr + batch_id * C_out * HW
    tl.store(out_base + offs_m[:, None] * HW + offs_n[None, :],
             acc, mask=mask_m[:, None] & mask_n[None, :])


# ============ Add Kernel ============
@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


# ============ Implementation helpers ============
def _triton_conv1x1(feat, weight, bias):
    B = feat.shape[0]
    C_in = feat.shape[1]
    H = feat.shape[2]
    W = feat.shape[3]
    C_out = weight.shape[0]
    HW = H * W

    conv_out = torch.empty(B, C_out, H, W, dtype=feat.dtype, device=feat.device)

    if feat.element_size() <= 2:
        BLOCK_N = 128
        BLOCK_K = 128
        n_stages = 4
        n_warps = 8
    else:
        BLOCK_N = 128
        BLOCK_K = 64
        n_stages = 3
        n_warps = 4

    num_n_blocks = (HW + BLOCK_N - 1) // BLOCK_N
    grid = (B * num_n_blocks,)

    conv1x1_kernel[grid](
        feat, weight, bias, conv_out,
        B, C_in, C_out, HW,
        BLOCK_M=32,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=n_warps,
        num_stages=n_stages,
    )
    return conv_out


def _fused_add_dropout(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


# ============ Shared dispatch wrapper ============
@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]
    if route == "conv":
        return _triton_conv1x1(args[0], args[1], args[2])
    elif route == "add":
        return _fused_add_dropout(args[0], args[1])