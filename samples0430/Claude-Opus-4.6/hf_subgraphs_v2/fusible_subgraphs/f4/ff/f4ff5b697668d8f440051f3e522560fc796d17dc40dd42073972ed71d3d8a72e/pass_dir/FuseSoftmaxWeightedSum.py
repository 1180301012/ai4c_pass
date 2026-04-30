import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """Match softmax + reshape + weighted multiply + sum + cat pattern.
    
    in_0: [1, 1, 1, 64] - linspace_x weights
    in_1: [1, 1, 64, 1] - linspace_y weights  
    in_2: [B, 17, 4096] - input to softmax
    """
    tmp_2 = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(128, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(128, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return (tmp_3, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=1),
    ],
    key=[],
)
@triton.jit
def fused_softmax_weighted_sum_kernel(
    in_2_ptr,
    in_0_ptr,
    in_1_ptr,
    out_softmax_ptr,
    out_coords_ptr,
    BK,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= BK:
        return

    # Each program handles one (batch, keypoint) pair = 4096 elements
    row_offset = pid * 4096
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load input row and convert to fp32 for numerical stability
    x = tl.load(in_2_ptr + row_offset + offsets).to(tl.float32)

    # Softmax computation
    max_val = tl.max(x, axis=0)
    x = x - max_val
    x = tl.exp(x)
    sum_val = tl.sum(x, axis=0)
    x = x / sum_val

    # Store softmax output (auto-converts to output dtype)
    tl.store(out_softmax_ptr + row_offset + offsets, x)

    # Load x-weights (in_0 has shape [1,1,1,64], stored as 64 contiguous values)
    # For flat index k in 0..4095, x-weight index = k % 64
    x_weights = tl.load(in_0_ptr + (offsets % 64)).to(tl.float32)
    
    # Load y-weights (in_1 has shape [1,1,64,1], stored as 64 contiguous values)
    # For flat index k in 0..4095, y-weight index = k // 64
    y_weights = tl.load(in_1_ptr + (offsets // 64)).to(tl.float32)

    # Compute weighted sums
    x_sum = tl.sum(x * x_weights, axis=0)
    y_sum = tl.sum(x * y_weights, axis=0)

    # Store coordinates [x_sum, y_sum] for this (batch, keypoint)
    tl.store(out_coords_ptr + pid * 2, x_sum)
    tl.store(out_coords_ptr + pid * 2 + 1, y_sum)


@torch.fx.wrap
def fused_softmax_weighted_sum(in_0, in_1, in_2):
    B = in_2.shape[0]
    K = 17
    BK = B * K

    # Ensure input is contiguous for correct pointer arithmetic
    in_2_contig = in_2.contiguous()

    # Allocate outputs
    out_softmax = torch.empty(B, K, 64, 64, dtype=in_2.dtype, device=in_2.device)
    out_coords = torch.empty(B, K, 2, dtype=in_2.dtype, device=in_2.device)

    # Launch kernel: one program per (batch, keypoint) pair
    grid = (BK,)
    fused_softmax_weighted_sum_kernel[grid](
        in_2_contig,
        in_0,
        in_1,
        out_softmax,
        out_coords,
        BK,
    )

    return out_softmax, out_coords


def replacement_func():
    return fused_softmax_weighted_sum