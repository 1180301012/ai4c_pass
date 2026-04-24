import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['total_per_batch'],
)
@triton.jit
def fused_view_cat_sigmoid_sub_mul_kernel(
    in3_ptr,
    in4_ptr,
    conv_ptr,
    out_ptr,
    L1,
    L2,
    conv_last,
    total_per_batch,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: out = (sigmoid(x) - 0.25) * pi
    where x is the concatenated tensor [in3 | in4 | conv_out] along dim 2.
    Grid: (B, ceil(total_per_batch / BLOCK_SIZE))
    x is loaded in native dtype, sigmoid computed in fp32, result cast back.
    """
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)

    block_start = pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_per_batch

    global_offsets = pid_batch * total_per_batch + offsets

    # Determine which source tensor each element belongs to
    in3_range = offsets < L1
    in4_range = (offsets >= L1) & (offsets < L1 + L2)

    # Load from in3 [B, 1, L1]
    in3_val = tl.load(in3_ptr + pid_batch * L1 + offsets, mask=in3_range, other=0.0)

    # Load from in4 [B, 1, L2]
    in4_val = tl.load(in4_ptr + pid_batch * L2 + (offsets - L1), mask=in4_range, other=0.0)

    # Load from conv_out [B, 1, conv_last]
    conv_val = tl.load(conv_ptr + pid_batch * conv_last + (offsets - L1 - L2),
                       mask=(~in3_range) & (~in4_range) & mask, other=0.0)

    # Merge: use the value from the correct tensor
    x = tl.where(in3_range, in3_val, tl.where(in4_range, in4_val, conv_val))

    # Cast to fp32 for sigmoid (only supported on fp32/fp64)
    x_fp32 = x.to(tl.float32)

    # Compute (sigmoid(x) - 0.25) * pi in fp32
    out_fp32 = tl.sigmoid(x_fp32) - 0.25
    out_fp32 = out_fp32 * 3.141592653589793

    # Cast back to original dtype and store
    out = out_fp32.to(x.dtype)
    tl.store(out_ptr + global_offsets, out, mask=mask)


@torch.fx.wrap
def fused_view_cat_sigmoid_sub_mul(conv_out, in_3, in_4, L1, L2, conv_last):
    """
    Fused view + cat + sigmoid + sub(0.25) + mul(pi) in a single Triton kernel.
    conv_out: [B, 1, conv_last]
    in_3:     [B, 1, L1]
    in_4:     [B, 1, L2]
    returns:  [B, 1, L1+L2+conv_last]  (single tensor, not tuple)
    """
    B = conv_out.shape[0]
    total = L1 + L2 + conv_last

    out = torch.empty((B, 1, total), dtype=conv_out.dtype, device=conv_out.device)

    def grid(meta):
        return (B, triton.cdiv(total, meta['BLOCK_SIZE']))

    fused_view_cat_sigmoid_sub_mul_kernel[grid](
        in_3, in_4, conv_out, out,
        L1, L2, conv_last, total,
    )

    return out