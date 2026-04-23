import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return (tmp_6,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 64}, num_warps=2),
        triton.Config({'BLOCK_H': 128}, num_warps=4),
        triton.Config({'BLOCK_H': 256}, num_warps=4),
        triton.Config({'BLOCK_H': 256}, num_warps=8),
        triton.Config({'BLOCK_H': 512}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def fused_masked_mean_kernel(
    in0_ptr, in1_ptr, out_ptr,
    batch_size, hidden_dim,
    stride_in0_b, stride_in0_s, stride_in0_h,
    stride_in1_b, stride_in1_s, stride_in1_h,
    stride_out_b, stride_out_h,
    SEQ_LEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    h_start = pid_h * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < hidden_dim

    masked_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    mask_sum = tl.zeros([BLOCK_H], dtype=tl.float32)

    for s in range(SEQ_LEN):
        # Load mask value (int64 -> float32)
        in0_ptrs = in0_ptr + pid_b * stride_in0_b + s * stride_in0_s + h_offsets * stride_in0_h
        in0_val = tl.load(in0_ptrs, mask=h_mask, other=0).to(tl.float32)

        # Load hidden states value (bfloat16/float16 -> float32)
        in1_ptrs = in1_ptr + pid_b * stride_in1_b + s * stride_in1_s + h_offsets * stride_in1_h
        in1_val = tl.load(in1_ptrs, mask=h_mask, other=0.0).to(tl.float32)

        # Accumulate masked sum and mask sum
        masked_sum += in1_val * in0_val
        mask_sum += in0_val

    # Clamp mask sum (min=1e-9) and divide
    clamped_mask_sum = tl.maximum(mask_sum, 1e-9)
    result = masked_sum / clamped_mask_sum

    # Store output
    out_ptrs = out_ptr + pid_b * stride_out_b + h_offsets * stride_out_h
    tl.store(out_ptrs, result, mask=h_mask)


@torch.fx.wrap
def fused_masked_mean(in_0, in_1):
    batch_size, seq_len, hidden_dim = in_0.shape

    # Output shape: [batch_size, hidden_dim], dtype float32
    out = torch.empty((batch_size, hidden_dim), dtype=torch.float32, device=in_0.device)

    grid = lambda meta: (batch_size, triton.cdiv(hidden_dim, meta['BLOCK_H']))

    fused_masked_mean_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        stride_in0_b=in_0.stride(0),
        stride_in0_s=in_0.stride(1),
        stride_in0_h=in_0.stride(2),
        stride_in1_b=in_1.stride(0),
        stride_in1_s=in_1.stride(1),
        stride_in1_h=in_1.stride(2),
        stride_out_b=out.stride(0),
        stride_out_h=out.stride(1),
        SEQ_LEN=seq_len,
    )

    return (out,)


def replacement_func():
    return fused_masked_mean