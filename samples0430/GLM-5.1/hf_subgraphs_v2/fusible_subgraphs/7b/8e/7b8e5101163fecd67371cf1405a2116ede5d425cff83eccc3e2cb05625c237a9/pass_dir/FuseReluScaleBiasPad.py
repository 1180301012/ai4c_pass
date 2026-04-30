import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace = False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 32}, num_warps=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 32}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64}, num_warps=8),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}, num_warps=8),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 64}, num_warps=16),
    ],
    key=['H', 'W'],
)
@triton.jit
def fused_relu_scale_bias_kernel(
    input_ptr, bias_ptr, scale_ptr, output_ptr,
    B, C, H, W,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    num_h_blocks = tl.cdiv(H, BLOCK_H)
    num_w_blocks = tl.cdiv(W, BLOCK_W)

    pid = tl.program_id(0)

    # Decompose pid into (b, c, h_block, w_block)
    w_block = pid % num_w_blocks
    pid_rest = pid // num_w_blocks
    h_block = pid_rest % num_h_blocks
    pid_rest = pid_rest // num_h_blocks
    c = pid_rest % C
    b = pid_rest // C

    h_start = h_block * BLOCK_H
    w_start = w_block * BLOCK_W

    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)

    # Masks for valid positions within [H, W] region
    h_valid = h_offsets < H
    w_valid = w_offsets < W

    # Load scale and bias (scalar tensors with shape [1])
    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)

    # 2D mask
    mask = h_valid[:, None] & w_valid[None, :]

    # Compute offsets using strides
    input_off = b * input_stride_b + c * input_stride_c + h_offsets[:, None] * input_stride_h + w_offsets[None, :] * input_stride_w
    output_off = b * output_stride_b + c * output_stride_c + h_offsets[:, None] * output_stride_h + w_offsets[None, :] * output_stride_w

    # Load input
    x = tl.load(input_ptr + input_off, mask=mask, other=0.0)

    # ReLU: max(x, 0)
    x_relu = tl.where(x > 0.0, x, 0.0)

    # Scale + bias
    result = x_relu * scale + bias

    # Store
    tl.store(output_ptr + output_off, result, mask=mask)

@torch.fx.wrap
def fused_relu_scale_bias_pad(in_0, in_1, in_2):
    B, C, H, W = in_2.shape
    H_out = H + 1
    W_out = W + 1

    BLOCK_H = 8
    BLOCK_W = 64

    num_h_blocks = triton.cdiv(H, BLOCK_H)
    num_w_blocks = triton.cdiv(W, BLOCK_W)
    total_programs = B * C * num_h_blocks * num_w_blocks

    # Allocate output initialized to zeros (padding region is automatically zero)
    output = torch.zeros((B, C, H_out, W_out), dtype=in_2.dtype, device=in_2.device)

    in_strides = in_2.stride()
    out_strides = output.stride()

    fused_relu_scale_bias_kernel[(total_programs,)](
        input_ptr=in_2, bias_ptr=in_0, scale_ptr=in_1, output_ptr=output,
        B=B, C=C, H=H, W=W,
        input_stride_b=in_strides[0], input_stride_c=in_strides[1],
        input_stride_h=in_strides[2], input_stride_w=in_strides[3],
        output_stride_b=out_strides[0], output_stride_c=out_strides[1],
        output_stride_h=out_strides[2], output_stride_w=out_strides[3],
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )

    return output

def replacement_func():
    return fused_relu_scale_bias_pad