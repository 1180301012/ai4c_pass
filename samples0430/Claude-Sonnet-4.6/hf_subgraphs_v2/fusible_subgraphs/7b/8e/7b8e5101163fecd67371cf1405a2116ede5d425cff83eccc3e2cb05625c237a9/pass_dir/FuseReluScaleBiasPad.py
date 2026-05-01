import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_relu_scale_bias_pad_kernel(
    in0_ptr,   # bias  [1]
    in1_ptr,   # scale [1]
    in2_ptr,   # input [N, C, H, W]
    out_ptr,   # output [N, C, H+1, W+1]
    N, C, H, W,
    H_out, W_out,   # H+1, W+1
    n_elements,     # N * C * H_out * W_out
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose flat output index → (n, c, h, w) in output layout
    w_idx  = offsets % W_out
    rest   = offsets // W_out
    h_idx  = rest % H_out
    rest2  = rest // H_out
    c_idx  = rest2 % C
    n_idx  = rest2 // C

    # Load scalar scale and bias, upcast to fp32 for computation
    scale = tl.load(in1_ptr).to(tl.float32)
    bias  = tl.load(in0_ptr).to(tl.float32)

    # Valid region: not in the padded border
    valid = (h_idx < H) & (w_idx < W)

    # Compute flat index into input [N, C, H, W]
    in_idx = n_idx * (C * H * W) + c_idx * (H * W) + h_idx * W + w_idx

    # Load input values; masked positions get 0 (padding → 0 output anyway)
    x = tl.load(in2_ptr + in_idx, mask=(valid & mask), other=0.0).to(tl.float32)

    # relu(x) * scale + bias
    relu_x = tl.where(x > 0.0, x, 0.0)
    result = relu_x * scale + bias

    # Padding positions → 0
    out_val = tl.where(valid, result, 0.0)

    tl.store(out_ptr + offsets, out_val, mask=mask)


@torch.fx.wrap
def fused_relu_scale_bias_pad(in_0, in_1, in_2):
    N, C, H, W = in_2.shape
    H_out = H + 1
    W_out = W + 1
    n_elements = N * C * H_out * W_out

    out = torch.empty((N, C, H_out, W_out), dtype=in_2.dtype, device=in_2.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _fused_relu_scale_bias_pad_kernel[grid](
        in_0, in_1, in_2,
        out,
        N, C, H, W,
        H_out, W_out,
        n_elements,
    )

    return out


def replacement_func():
    return fused_relu_scale_bias_pad