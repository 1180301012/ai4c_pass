"""
Pass: FuseConvMaxpool_7x7_s2p3
Matches max_pool2d(kernel=3, stride=2, padding=1, dilation=1, ceil_mode=False)
and replaces it with a Triton kernel.

The conv2d node stays in the FX graph unchanged (cuDNN handles it).
Only the max_pool2d node is replaced.  This matches both resnetv2_101 graphs.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: 3x3 max-pool, stride=2, padding=1, dilation=1
# Grid: (N*C, H_out, ceil(W_out / BLOCK_W))
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 32}, num_warps=4),
        triton.Config({'BLOCK_W': 64}, num_warps=4),
        triton.Config({'BLOCK_W': 128}, num_warps=8),
    ],
    key=['N', 'C', 'H_out', 'W_out'],
)
@triton.jit
def triton_max_pool2d_3x3_s2p1_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    s_n, s_c, s_h, s_w,        # input strides
    os_n, os_c, os_h, os_w,    # output strides
    BLOCK_W: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h  = tl.program_id(1)
    pid_w  = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc %  C
    h_out = pid_h

    w_start = pid_w * BLOCK_W
    w_offs  = w_start + tl.arange(0, BLOCK_W)
    w_mask  = w_offs < W_out

    # stride=2, padding=1 → top-left input coord = out_coord*2 - 1
    h_in_base = h_out * 2 - 1
    w_in_base = w_offs * 2 - 1

    NEG = -65504.0
    max_val = tl.full([BLOCK_W], NEG, dtype=tl.float32)

    for kh in tl.static_range(3):
        h_in = h_in_base + kh
        h_ok = (h_in >= 0) & (h_in < H_in)
        for kw in tl.static_range(3):
            w_in  = w_in_base + kw
            valid = h_ok & (w_in >= 0) & (w_in < W_in) & w_mask
            in_off = n * s_n + c * s_c + h_in * s_h + w_in * s_w
            val = tl.load(input_ptr + in_off, mask=valid, other=NEG).to(tl.float32)
            max_val = tl.maximum(max_val, val)

    out_val = max_val.to(DTYPE)
    out_off = n * os_n + c * os_c + h_out * os_h + w_offs * os_w
    tl.store(output_ptr + out_off, out_val, mask=w_mask)


# ---------------------------------------------------------------------------
# Pattern: match ONLY the max_pool2d node (conv2d stays in the graph).
# Parameters: kernel=3, stride=2, padding=1, dilation=1, ceil_mode=False
# ---------------------------------------------------------------------------
def pattern(x):
    return torch.nn.functional.max_pool2d(x, 3, 2, 1, 1, ceil_mode=False, return_indices=False)


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# dtype mapping (module-level, no blocked calls)
# ---------------------------------------------------------------------------
_DTYPE_MAP = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


# ---------------------------------------------------------------------------
# Replacement: Triton max-pool only (conv remains in the FX graph)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_maxpool_3x3_s2p1(x):
    """
    x : input tensor [N, C, H_in, W_in]  (output of conv2d in the graph)
    """
    x = x.contiguous()
    N, C, H_in, W_in = x.shape
    H_out = (H_in - 1) // 2 + 1
    W_out = (W_in - 1) // 2 + 1

    output = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    DTYPE = _DTYPE_MAP.get(x.dtype, tl.float32)

    grid = lambda meta: (N * C, H_out, triton.cdiv(W_out, meta['BLOCK_W']))
    triton_max_pool2d_3x3_s2p1_kernel[grid](
        x, output,
        N, C, H_in, W_in, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        DTYPE=DTYPE,
    )
    return output


def replacement_func():
    return triton_maxpool_3x3_s2p1