import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return (tmp_3, tmp_1)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def combined_kernel(
    in0_ptr,
    in1_ptr,
    out0_ptr,
    out1_ptr,
    KHW: tl.constexpr,
    HW: tl.constexpr,
    W_dim: tl.constexpr,
    H_dim: tl.constexpr,
    C_dim: tl.constexpr,
    K_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_EXPAND: tl.constexpr,
):
    c_idx = tl.program_id(0)

    # Part 1: sum+div on in_1 for this channel
    h_range = tl.arange(0, BLOCK_H)

    for w_idx in range(W_dim):
        offsets = c_idx * HW + h_range * W_dim + w_idx
        vals = tl.load(in1_ptr + offsets)
        sum_val = tl.sum(vals)
        result = vals / sum_val
        tl.store(out1_ptr + offsets, result)

    # Part 2: expand on in_0 for this channel
    # output[b,c,k,h,w] = input[b,c,h,w] for all k
    # flat: input_offset = c * HW + offset % HW
    for block_start in range(0, KHW, BLOCK_EXPAND):
        expand_offsets = c_idx * KHW + block_start + tl.arange(0, BLOCK_EXPAND)
        input_offsets = c_idx * HW + expand_offsets % HW
        vals = tl.load(in0_ptr + input_offsets)
        tl.store(out0_ptr + expand_offsets, vals)


@torch.fx.wrap
def combined_forward(in_0, in_1):
    B, C, H, W = in_0.shape
    K = 64  # expand broadcast dimension

    # Create output tensors
    out1 = torch.empty_like(in_1)  # tmp_1: [1, 2, 8, 8]
    out0 = torch.empty(B, C, K, H, W, dtype=in_0.dtype, device=in_0.device)  # tmp_3: [1, 2, 64, 8, 8]

    HW = H * W
    KHW = K * H * W
    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_EXPAND = 512

    grid = (B * C,)  # 2 programs for B=1, C=2

    combined_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out0_ptr=out0,
        out1_ptr=out1,
        KHW=KHW,
        HW=HW,
        W_dim=W,
        H_dim=H,
        C_dim=C,
        K_dim=K,
        BLOCK_H=BLOCK_H,
        BLOCK_EXPAND=BLOCK_EXPAND,
    )

    return (out0, out1)


def replacement_func():
    return combined_forward