import operator
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused kernel: depthwise grouped conv1d (kernel=65, same-padding=32)
#               + in-place-add + permute(0,2,1,3) + contiguous write
# Input layouts:  in2 [B, G, S, C], in1 [B, G, S, C], weight [G, 1, 65, 1]
# Output layout:  [B, S, G, C]   (contiguous; view to [B,S,G*C] is free)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 8}),
        triton.Config({'BLOCK_C': 16}),
        triton.Config({'BLOCK_C': 32}),
        triton.Config({'BLOCK_C': 64}),
        triton.Config({'BLOCK_C': 128}),
    ],
    key=['C', 'OUTPUT_DTYPE'],
)
@triton.jit
def fused_dw_conv_add_transpose_g12_kernel(
    in2_ptr, in1_ptr, weight_ptr, out_ptr,
    B, G, S, C,
    OUTPUT_DTYPE: tl.constexpr,   # 0=float32, 1=float16, 2=bfloat16
    BLOCK_C: tl.constexpr,
):
    # one program per (b, s, g) triple
    pid = tl.program_id(0)
    g_idx = pid % G
    tmp   = pid // G
    s_idx = tmp % S
    b_idx = tmp // S

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # Strides for [B, G, S, C] contiguous layout
    bgs_stride = G * S * C
    gs_stride  = S * C

    # Accumulate conv in fp32 for precision
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Kernel size = 65, half = 32  →  same-size output
    for k in range(65):
        s_in = s_idx + k - 32
        valid = (s_in >= 0) & (s_in < S)

        # weight[g, 0, k, 0]  →  flat index = g*65 + k
        w = tl.load(weight_ptr + g_idx * 65 + k).to(tl.float32)

        in2_off = b_idx * bgs_stride + g_idx * gs_stride + s_in * C
        vals = tl.load(in2_ptr + in2_off + c_offs,
                       mask=c_mask & valid, other=0.0).to(tl.float32)
        acc += w * vals

    # Add in1[b, g, s, :]
    in1_off = b_idx * bgs_stride + g_idx * gs_stride + s_idx * C
    in1_vals = tl.load(in1_ptr + in1_off + c_offs, mask=c_mask).to(tl.float32)
    result = acc + in1_vals

    # Write to output in [B, S, G, C] layout
    out_off = b_idx * (S * G * C) + s_idx * (G * C) + g_idx * C
    if OUTPUT_DTYPE == 1:
        tl.store(out_ptr + out_off + c_offs, result.to(tl.float16),  mask=c_mask)
    elif OUTPUT_DTYPE == 2:
        tl.store(out_ptr + out_off + c_offs, result.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(out_ptr + out_off + c_offs, result.to(tl.float32),  mask=c_mask)


_DTYPE_MAP_G12 = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


@torch.fx.wrap
def fused_conv_add_transpose_g12(in_0, in_1, in_2):
    B, G, S, C = in_1.shape
    output = torch.empty(B, S, G, C, dtype=in_1.dtype, device=in_1.device)
    dtype_code = _DTYPE_MAP_G12.get(in_1.dtype, 0)

    grid = (B * S * G,)
    fused_dw_conv_add_transpose_g12_kernel[grid](
        in_2, in_1, in_0, output,
        B, G, S, C,
        OUTPUT_DTYPE=dtype_code,
    )
    return output


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 12)
    in_1 = operator.iadd(in_1, conv2d)
    tmp_3 = in_1.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv_add_transpose_g12