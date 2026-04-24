"""
Shared depthwise conv Triton kernel + single @torch.fx.wrap dispatch function.
All 5 conv passes import `dispatch_dw_conv` from here so they share the SAME
function object → satisfies output_pass_replacement_func_limit=1.
The route string selects stride/groups at runtime.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 64}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C', 'H_in', 'W_in', 'H_out', 'W_out'],
)
@triton.jit
def _dw_conv_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    stride_h, stride_w,
    BLOCK_W: tl.constexpr,
):
    nc       = tl.program_id(0)
    h_out_id = tl.program_id(1)
    w_block  = tl.program_id(2)
    n = nc // C;  c = nc % C
    w_offs = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W_out
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    for kh in tl.static_range(3):
        h_in = h_out_id + kh - 1
        h_ok = (h_in >= 0) & (h_in < H_in)
        for kw in tl.static_range(3):
            w_in = w_offs * stride_w + kw - 1
            valid = h_ok & (w_in >= 0) & (w_in < W_in) & w_mask
            x_idx = ((n * C + c) * H_in + h_in) * W_in + w_in
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)
            w_idx = c * 9 + kh * 3 + kw
            w_val = tl.load(w_ptr + w_idx).to(tl.float32)
            acc += x_val * w_val
    out_base = ((n * C + c) * H_out + h_out_id) * W_out
    tl.store(out_ptr + out_base + w_offs, acc, mask=w_mask)


@torch.fx.wrap
def dispatch_dw_conv(input, weight, route):
    """
    Single @torch.fx.wrap dispatch shared by all 5 conv passes.
    Route string selects stride and output spatial size:
      's1_c384', 's1_c768', 's1_c256' → stride=1, H_out=H_in, W_out=W_in
      's2_c384', 's2_256', 's2_c768' → stride=2, H_out=(H+1)//2, W_out=(W+1)//2
    """
    N, C, H_in, W_in = input.shape
    device = input.device
    x = input.to(device)
    w = weight.to(device)

    if route == "s1_c384":
        H_out = H_in; W_out = W_in
    elif route == "s1_c768":
        H_out = H_in; W_out = W_in
    elif route == "s1_c256":
        H_out = H_in; W_out = W_in
    elif route == "s2_c384":
        H_out = (H_in + 1) // 2; W_out = (W_in + 1) // 2
    elif route == "s2_c256":
        H_out = (H_in + 1) // 2; W_out = (W_in + 1) // 2
    elif route == "s2_c768":
        H_out = (H_in + 1) // 2; W_out = (W_in + 1) // 2
    else:
        H_out = H_in; W_out = W_in

    out  = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=device)
    grid = lambda meta: (N * C, H_out, triton.cdiv(W_out, meta['BLOCK_W']))
    _dw_conv_kernel[grid](x, w, out, N, C, H_in, W_in, H_out, W_out,
                          1 if route in ("s1_c384", "s1_c768", "s1_c256") else 2,
                          1 if route in ("s1_c384", "s1_c768", "s1_c256") else 2)
    return out