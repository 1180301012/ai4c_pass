import math
import torch
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def _conv1x1_reduce_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    logits_ptr,
    B,
    C,
    S,
    stride_bs,
    stride_c,
    stride_s,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    acc = tl.zeros((1,), dtype=tl.float32)
    offs_c = tl.arange(0, BLOCK_C)
    base = x_ptr + b * stride_bs + s * stride_s
    for c_start in range(0, C, BLOCK_C):
        c_idx = c_start + offs_c
        mask = c_idx < C
        x = tl.load(base + c_idx * stride_c, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * w, axis=0)

    bias = tl.load(b_ptr).to(tl.float32)
    out = acc + bias
    tl.store(logits_ptr + pid, out)


@triton.jit
def _softmax_rows_kernel(
    logits_ptr,
    out_ptr,
    B,
    S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    row_ptr = logits_ptr + pid * S + offs
    x = tl.load(row_ptr, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)
    x_max = tl.max(x, axis=0)
    num = tl.exp(x - x_max)
    den = tl.sum(num, axis=0)
    y = num / den
    out_row_ptr = out_ptr + pid * S + offs
    tl.store(out_row_ptr, y, mask=mask)


@torch.fx.wrap
def fused_conv1x1_softmax_unsqueeze(in_0, in_1, in_2):
    # Match only the intended single-output-channel 1x1 conv family.
    B = in_2.shape[0]
    C = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    S = H * W

    assert in_0.numel() == 1
    assert in_1.ndim == 4
    assert in_1.shape[0] == 1 and in_1.shape[2] == 1 and in_1.shape[3] == 1
    assert in_1.shape[1] == C

    # Flatten spatial dimensions with a view-like layout assumption preserved by these models.
    x = in_2.reshape(B, C, S)
    x = x.contiguous()
    w = in_1.reshape(C).contiguous()
    b = in_0.reshape(1).contiguous()

    logits = torch.empty((B, S), device=in_2.device, dtype=torch.float32)
    out = torch.empty((B, 1, S, 1), device=in_2.device, dtype=in_2.dtype)

    # Choose moderately large reduction tiles; C is small/moderate (80/304/512/608).
    BLOCK_C = 128 if C <= 128 else 256
    grid_reduce = (B * S,)
    _conv1x1_reduce_kernel[grid_reduce](
        x,
        w,
        b,
        logits,
        B,
        C,
        S,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        BLOCK_C=BLOCK_C,
    )

    BLOCK_S = min(4096, _next_power_of_2(S))
    grid_softmax = (B,)
    _softmax_rows_kernel[grid_softmax](
        logits,
        out,
        B,
        S,
        BLOCK_S=BLOCK_S,
    )
    return out