import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    in_1 += in_0
    tmp_1 = in_1.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(in_1)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_add_softmax_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    n_rows, seq_len,
    in_0_row_stride, in_1_row_stride, out_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < seq_len

    in_0_start = row_idx * in_0_row_stride
    in_1_start = row_idx * in_1_row_stride
    out_start = row_idx * out_row_stride

    # Load input values (out-of-bounds padded with 0 so add is neutral)
    x0 = tl.load(in_0_ptr + in_0_start + cols, mask=mask, other=0.0)
    x1 = tl.load(in_1_ptr + in_1_start + cols, mask=mask, other=0.0)

    # Add inputs, then upcast to float32 for numerically stable softmax
    x_f32 = (x0 + x1).to(tl.float32)

    # Force out-of-bounds lanes to -inf so they contribute 0 to the softmax sum
    NEG_INF = float('-inf')
    x_f32 = tl.where(mask, x_f32, NEG_INF)

    # Numerically stable softmax: subtract max before exp
    x_max = tl.max(x_f32, axis=0)
    x_shifted = x_f32 - x_max
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, axis=0)
    softmax_f32 = x_exp / x_sum

    # Cast back to the original input dtype (fp16, bf16, or fp32)
    softmax_out = softmax_f32.to(x0.dtype)

    # Store result (only valid lanes)
    tl.store(out_ptr + out_start + cols, softmax_out, mask=mask)


@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    seq_len = in_0.shape[-1]
    n_rows = in_0.numel() // seq_len

    out = torch.empty_like(in_0)

    fused_add_softmax_kernel[n_rows,](
        in_0, in_1, out,
        n_rows, seq_len,
        in_0.stride(-2), in_1.stride(-2), out.stride(-2),
    )

    return out


def replacement_func():
    return fused_add_softmax