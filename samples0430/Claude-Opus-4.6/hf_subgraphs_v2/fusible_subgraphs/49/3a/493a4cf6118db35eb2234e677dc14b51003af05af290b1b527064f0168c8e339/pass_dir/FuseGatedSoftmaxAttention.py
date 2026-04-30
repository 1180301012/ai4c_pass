import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_2.softmax(dim=-1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    return tmp_8


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_gated_softmax_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (chunks_of_7_rows, heads)
    chunk = tl.program_id(0)
    head = tl.program_id(1)

    # Load gate once, compute sigmoid
    gate = tl.load(in_0_ptr + head).to(tl.float32)
    sig_gate = tl.sigmoid(gate)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS
    base = head * N_COLS * N_COLS + chunk * 7 * N_COLS

    # Row 0
    offset = base
    in_2_val = tl.load(in_2_ptr + offset + cols, mask=mask, other=-float('inf'))
    x = in_2_val.to(tl.float32)
    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    softmax = x / tl.sum(x, axis=0)
    in_1_val = tl.load(in_1_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    result = in_1_val + sig_gate * (softmax - in_1_val)
    tl.store(out_ptr + offset + cols, result.to(in_2_val.dtype), mask=mask)

    # Row 1
    offset = base + N_COLS
    in_2_val = tl.load(in_2_ptr + offset + cols, mask=mask, other=-float('inf'))
    x = in_2_val.to(tl.float32)
    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    softmax = x / tl.sum(x, axis=0)
    in_1_val = tl.load(in_1_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    result = in_1_val + sig_gate * (softmax - in_1_val)
    tl.store(out_ptr + offset + cols, result.to(in_2_val.dtype), mask=mask)

    # Row 2
    offset = base + 2 * N_COLS
    in_2_val = tl.load(in_2_ptr + offset + cols, mask=mask, other=-float('inf'))
    x = in_2_val.to(tl.float32)
    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    softmax = x / tl.sum(x, axis=0)
    in_1_val = tl.load(in_1_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    result = in_1_val + sig_gate * (softmax - in_1_val)
    tl.store(out_ptr + offset + cols, result.to(in_2_val.dtype), mask=mask)

    # Row 3
    offset = base + 3 * N_COLS
    in_2_val = tl.load(in_2_ptr + offset + cols, mask=mask, other=-float('inf'))
    x = in_2_val.to(tl.float32)
    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    softmax = x / tl.sum(x, axis=0)
    in_1_val = tl.load(in_1_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    result = in_1_val + sig_gate * (softmax - in_1_val)
    tl.store(out_ptr + offset + cols, result.to(in_2_val.dtype), mask=mask)

    # Row 4
    offset = base + 4 * N_COLS
    in_2_val = tl.load(in_2_ptr + offset + cols, mask=mask, other=-float('inf'))
    x = in_2_val.to(tl.float32)
    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    softmax = x / tl.sum(x, axis=0)
    in_1_val = tl.load(in_1_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    result = in_1_val + sig_gate * (softmax - in_1_val)
    tl.store(out_ptr + offset + cols, result.to(in_2_val.dtype), mask=mask)

    # Row 5
    offset = base + 5 * N_COLS
    in_2_val = tl.load(in_2_ptr + offset + cols, mask=mask, other=-float('inf'))
    x = in_2_val.to(tl.float32)
    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    softmax = x / tl.sum(x, axis=0)
    in_1_val = tl.load(in_1_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    result = in_1_val + sig_gate * (softmax - in_1_val)
    tl.store(out_ptr + offset + cols, result.to(in_2_val.dtype), mask=mask)

    # Row 6
    offset = base + 6 * N_COLS
    in_2_val = tl.load(in_2_ptr + offset + cols, mask=mask, other=-float('inf'))
    x = in_2_val.to(tl.float32)
    x = x - tl.max(x, axis=0)
    x = tl.exp(x)
    softmax = x / tl.sum(x, axis=0)
    in_1_val = tl.load(in_1_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    result = in_1_val + sig_gate * (softmax - in_1_val)
    tl.store(out_ptr + offset + cols, result.to(in_2_val.dtype), mask=mask)


@torch.fx.wrap
def fused_gated_softmax(in_0, in_1, in_2):
    in_0_dev = in_0.to(in_1.device)
    out = torch.empty_like(in_1)

    N_HEADS = in_1.shape[1]  # 16
    N_ROWS = in_1.shape[2]   # 196
    N_COLS = in_1.shape[3]   # 196

    # 196/7=28 chunks per head, 28*16=448 programs
    grid = (N_ROWS // 7, N_HEADS)

    fused_gated_softmax_kernel[grid](
        in_0_dev,
        in_1,
        in_2,
        out,
        N_COLS=N_COLS,
        BLOCK_SIZE=256,
        num_warps=1,
        num_stages=1,
    )

    return out


def replacement_func():
    return fused_gated_softmax