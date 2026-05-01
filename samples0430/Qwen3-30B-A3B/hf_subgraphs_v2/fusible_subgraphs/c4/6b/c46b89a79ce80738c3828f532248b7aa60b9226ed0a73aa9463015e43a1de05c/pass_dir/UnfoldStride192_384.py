import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = torch.nn.functional.unfold(x, kernel_size=(384, 384), stride=(192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def unfold_stride192_kernel(
    x_ptr,
    out_ptr,
    H,
    W,
    KH,
    KW,
    stride,
    C,
    out_h,
    out_w,
    block_size: tl.constexpr,
):
    idx = tl.program_id(0) * block_size + tl.arange(0, block_size)
    total_elements = out_h * out_w * C * KH * KW
    mask = idx < total_elements
    p = idx // (C * KH * KW)
    c = (idx // (KH * KW)) % C
    i = (idx // KW) % KH
    j = idx % KW
    base_row = (p // out_w) * stride
    base_col = (p % out_w) * stride
    input_row = base_row + i
    input_col = base_col + j
    offset = c * H * W + input_row * W + input_col
    x_val = tl.load(x_ptr + offset, mask=mask)
    tl.store(out_ptr + idx, x_val, mask=mask)

@torch.fx.wrap
def unfold_stride192(x):
    H = 768
    W = 768
    KH = 384
    KW = 384
    stride = 192
    out_h = (H - KH) // stride + 1
    out_w = (W - KW) // stride + 1
    P = out_h * out_w
    total_elements = P * 3 * KH * KW
    out = torch.empty(total_elements, dtype=x.dtype, device=x.device)
    block_size = 1024
    num_blocks = (total_elements + block_size - 1) // block_size
    unfold_stride192_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        H=H,
        W=W,
        KH=KH,
        KW=KW,
        stride=stride,
        C=3,
        out_h=out_h,
        out_w=out_w,
        block_size=block_size,
    )
    return out.reshape(P, 3, KH, KW)

def replacement_func():
    return unfold_stride192