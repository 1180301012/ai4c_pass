import torch
import triton
import triton.language as tl
import operator
import inspect


def pattern(in_0=None, in_1=None, in_2=None):
    """Build pattern graph manually with operator.iadd node."""
    graph = torch.fx.Graph()
    p_in_0 = graph.placeholder('in_0')
    p_in_1 = graph.placeholder('in_1')
    p_in_2 = graph.placeholder('in_2')
    conv2d = graph.call_function(torch.conv2d, (p_in_2, p_in_0, None, (1, 1), (32, 0), (1, 1), 12))
    iadd_result = graph.call_function(operator.iadd, (p_in_1, conv2d))
    permute = graph.call_method('permute', (iadd_result, 0, 2, 1, 3))
    contiguous = graph.call_method('contiguous', (permute,))
    graph.output(contiguous)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    gm.__signature__ = inspect.Signature(parameters=[
        inspect.Parameter('in_0', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_1', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_2', inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    return gm


pattern = pattern()


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "g12")


@triton.jit
def fused_conv_add_permute_kernel(
    in_2_ptr, in_0_ptr, in_1_ptr, out_ptr,
    B, G, H, W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    K: tl.constexpr,
    PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    num_h_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_block_idx = pid % num_h_blocks
    bg = pid // num_h_blocks
    g = bg % G
    b = bg // G

    h_start = h_block_idx * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)[:, None]
    w_offsets = tl.arange(0, BLOCK_W)[None, :]

    h_mask = h_offsets < H
    w_mask = w_offsets < W
    hw_mask = h_mask & w_mask

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    base = b * G * H * W + g * H * W
    weight_base = g * K

    for k in range(K):
        h_in = h_offsets + k - PAD
        in_mask = (h_in >= 0) & (h_in < H) & w_mask
        in_idx = base + h_in * W + w_offsets
        x = tl.load(in_2_ptr + in_idx, mask=in_mask, other=0.0)
        wt = tl.load(in_0_ptr + weight_base + k)
        acc += x * wt

    res_idx = base + h_offsets * W + w_offsets
    res = tl.load(in_1_ptr + res_idx, mask=hw_mask, other=0.0)
    result = res + acc

    GW = G * W
    out_idx = b * H * GW + h_offsets * GW + g * W + w_offsets
    tl.store(out_ptr + out_idx, result.to(out_ptr.dtype.element_ty), mask=hw_mask)


def _run_kernel(in_0, in_1, in_2):
    B, G, H, W = in_2.shape
    K = in_0.shape[2]
    PAD = K // 2

    out = torch.empty(B, H, G, W, dtype=in_1.dtype, device=in_1.device)

    if H >= 64:
        BLOCK_H = 64
    else:
        BLOCK_H = 1
        while BLOCK_H < H:
            BLOCK_H *= 2

    BLOCK_W = 1
    while BLOCK_W < W:
        BLOCK_W *= 2

    num_h_blocks = (H + BLOCK_H - 1) // BLOCK_H
    grid = (B * G * num_h_blocks,)

    fused_conv_add_permute_kernel[grid](
        in_2, in_0, in_1, out,
        B, G, H, W,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        K=K,
        PAD=PAD,
    )

    return out


@torch.fx.wrap
def fused_conv_add_permute_dispatch(in_0, in_1, in_2, route):
    return _run_kernel(in_0, in_1, in_2)


def replacement_func():
    return fused_conv_add_permute_dispatch