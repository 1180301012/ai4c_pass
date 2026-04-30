import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = in_0 * in_2
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return (tmp_2, tmp_13)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def rmsnorm_kernel(
    in_0_ptr,
    in_1_ptr,
    out_scaled_ptr,
    out_norm_ptr,
    normalizer_val: tl.constexpr,
    n_rows,
    n_cols,
    stride_in0_row,
    stride_out_scaled_row,
    stride_out_norm_row,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Compute row start pointers
    in_0_row_ptr = in_0_ptr + row_idx * stride_in0_row
    out_scaled_row_ptr = out_scaled_ptr + row_idx * stride_out_scaled_row
    out_norm_row_ptr = out_norm_ptr + row_idx * stride_out_norm_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load input as bfloat16, convert to float32 for computation
    x_bf16 = tl.load(in_0_row_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x_bf16.to(tl.float32)

    # Scale by normalizer (tmp_2 = in_0 * in_2)
    x_scaled = x_f32 * normalizer_val

    # Store scaled output as bfloat16 (tmp_2)
    tl.store(out_scaled_row_ptr + offsets, x_scaled.to(tl.bfloat16), mask=mask)

    # Compute mean of squares: tmp_5 = tmp_4.pow(2), tmp_6 = tmp_5.mean(-1, keepdim=True)
    x_sq = x_scaled * x_scaled
    mean_sq = tl.sum(x_sq, axis=0) / n_cols

    # Compute rsqrt(mean + eps): tmp_7 = tmp_6 + 1e-06, tmp_8 = torch.rsqrt(tmp_7)
    rms_inv = tl.rsqrt(mean_sq + eps)

    # Load weight: tmp_10 = in_1.float()
    w_bf16 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    w_f32 = w_bf16.to(tl.float32)

    # Compute gain and normalize: tmp_11 = 1.0 + tmp_10, tmp_9 = tmp_4 * tmp_8, tmp_12 = tmp_9 * tmp_11
    gain = 1.0 + w_f32
    normalized = x_scaled * rms_inv * gain

    # Store normalized output as bfloat16 (tmp_13 = tmp_12.type_as(tmp_2))
    tl.store(out_norm_row_ptr + offsets, normalized.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def rmsnorm_fused(in_0, in_1, in_2):
    # Extract scalar normalizer value (in_2 is a 0-dim bfloat16 tensor, possibly on CPU)
    normalizer_val = in_2.item()

    # Get dimensions - flatten to [num_rows, hidden_size]
    orig_shape = in_0.shape
    n_cols = orig_shape[-1]
    n_rows = in_0.numel() // n_cols

    # Ensure contiguous layout
    in_0_cont = in_0.contiguous()
    in_1_cont = in_1.contiguous()

    # Allocate output tensors with original shape and bfloat16 dtype
    out_scaled = torch.empty(orig_shape, dtype=in_0.dtype, device=in_0.device)
    out_norm = torch.empty(orig_shape, dtype=in_0.dtype, device=in_0.device)

    # Get strides for 2D view
    stride_in0 = in_0_cont.reshape(-1, n_cols).stride(0)
    stride_out_scaled = out_scaled.reshape(-1, n_cols).stride(0)
    stride_out_norm = out_norm.reshape(-1, n_cols).stride(0)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)

    rmsnorm_kernel[grid](
        in_0_ptr=in_0_cont.reshape(-1, n_cols),
        in_1_ptr=in_1_cont,
        out_scaled_ptr=out_scaled.reshape(-1, n_cols),
        out_norm_ptr=out_norm.reshape(-1, n_cols),
        normalizer_val=normalizer_val,
        n_rows=n_rows,
        n_cols=n_cols,
        stride_in0_row=stride_in0,
        stride_out_scaled_row=stride_out_scaled,
        stride_out_norm_row=stride_out_norm,
        eps=1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out_scaled, out_norm)


def replacement_func():
    return rmsnorm_fused