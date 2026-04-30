import torch
import triton
import triton.language as tl


def pattern(conv_out):
    tmp_2 = torch.nn.functional.pad(conv_out, [2, 2, 2, 2], 'constant', None)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 80, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    split = torch.functional.split(tmp_6, [16, 64], dim=-1)
    tmp_8 = split[0]
    tmp_9 = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return (tmp_10, tmp_9)


def replacement_args(conv_out):
    return (conv_out, "route_c640")


@triton.jit
def scatter_out1_kernel(
    conv_out_ptr, out1_ptr,
    cpg: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    out1 shape: [8, 4, 16, 144] in C-contiguous order
    out1[g, s, ch, p] = conv_out[g*cpg+ch, h*16+w] if valid else 0
    where h = (s//2)*8 + (p//12) - 2, w = (s%2)*8 + (p%12) - 2
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Decode flat index for [8, 4, 16, 144]
    p = offs % 144
    tmp = offs // 144
    ch = tmp % 16
    tmp2 = tmp // 16
    s = tmp2 % 4
    g = tmp2 // 4

    # Compute source position in conv_out [C_out, 256] (flattened spatial)
    c = g * cpg + ch
    i = s // 2
    j = s % 2
    kh = p // 12
    kw = p % 12
    h = i * 8 + kh - 2
    w = j * 8 + kw - 2

    valid = (h >= 0) & (h < 16) & (w >= 0) & (w < 16)
    src_idx = c * 256 + h * 16 + w

    val = tl.load(conv_out_ptr + src_idx, mask=mask & valid, other=0.0)
    tl.store(out1_ptr + offs, val, mask=mask)


@triton.jit
def scatter_out2_kernel(
    conv_out_ptr, out2_ptr,
    cpg: tl.constexpr,
    split2: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    out2 shape: [8, 4, 144, split2] in C-contiguous order
    out2[g, s, p, ch] = conv_out[g*cpg+ch+16, h*16+w] if valid else 0
    where h = (s//2)*8 + (p//12) - 2, w = (s%2)*8 + (p%12) - 2
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Decode flat index for [8, 4, 144, split2]
    ch = offs % split2
    tmp = offs // split2
    p = tmp % 144
    tmp2 = tmp // 144
    s = tmp2 % 4
    g = tmp2 // 4

    # Compute source position
    c = g * cpg + ch + 16
    i = s // 2
    j = s % 2
    kh = p // 12
    kw = p % 12
    h = i * 8 + kh - 2
    w = j * 8 + kw - 2

    valid = (h >= 0) & (h < 16) & (w >= 0) & (w < 16)
    src_idx = c * 256 + h * 16 + w

    val = tl.load(conv_out_ptr + src_idx, mask=mask & valid, other=0.0)
    tl.store(out2_ptr + offs, val, mask=mask)


@torch.fx.wrap
def dispatch_scatter(conv_out, route):
    if route == "route_c640":
        cpg = 80
        split2 = 64
    elif route == "route_c384":
        cpg = 48
        split2 = 32
    else:
        cpg = 48
        split2 = 32

    BLOCK_SIZE = 1024

    # Scatter to out1: [8, 4, 16, 144]
    n1 = 8 * 4 * 16 * 144
    out1 = torch.empty((8, 4, 16, 144), dtype=conv_out.dtype, device=conv_out.device)
    grid1 = ((n1 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    scatter_out1_kernel[grid1](conv_out, out1, cpg, n1, BLOCK_SIZE)

    # Scatter to out2: [8, 4, 144, split2]
    n2 = 8 * 4 * 144 * split2
    out2 = torch.empty((8, 4, 144, split2), dtype=conv_out.dtype, device=conv_out.device)
    grid2 = ((n2 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    scatter_out2_kernel[grid2](conv_out, out2, cpg, split2, n2, BLOCK_SIZE)

    return (out1, out2)


def replacement_func():
    return dispatch_scatter