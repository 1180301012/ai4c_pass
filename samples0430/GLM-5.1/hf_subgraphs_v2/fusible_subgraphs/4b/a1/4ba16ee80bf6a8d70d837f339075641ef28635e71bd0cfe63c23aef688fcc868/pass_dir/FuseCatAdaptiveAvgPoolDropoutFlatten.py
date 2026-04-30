import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return (tmp_3,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_2d_pool_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    C0, C1, C2, C3,
    HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    input_id = tl.program_id(0)  # 0, 1, 2, 3 - selects which input tensor
    channel_block = tl.program_id(1)  # which block of channels within that input

    # Select the appropriate input pointer and compute offset based on input_id
    # Program-level branching - all threads in this program take the same path
    if input_id == 0:
        in_ptr = in0_ptr
        C_in = C0
        offset = 0
    elif input_id == 1:
        in_ptr = in1_ptr
        C_in = C1
        offset = C0
    elif input_id == 2:
        in_ptr = in2_ptr
        C_in = C2
        offset = C0 + C1
    else:
        in_ptr = in3_ptr
        C_in = C3
        offset = C0 + C1 + C2

    c_start = channel_block * BLOCK_C
    c_off = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_off < C_in

    hw_off = tl.arange(0, BLOCK_HW)
    hw_mask = hw_off < HW

    # Load from the single selected input - no conditional logic needed
    ptr = in_ptr + c_off[:, None] * HW + hw_off[None, :]
    vals = tl.load(ptr, mask=c_mask[:, None] & hw_mask[None, :], other=0.0)

    # Sum over spatial dimension (accumulate in float32 for precision)
    sum_vals = tl.sum(vals.to(tl.float32), axis=1)
    mean_vals = sum_vals / HW

    # Store to output at correct channel offset
    out_off = c_off + offset
    tl.store(out_ptr + out_off, mean_vals, mask=c_mask)


@torch.fx.wrap
def fused_cat_pool_flatten(in_0, in_1, in_2, in_3):
    C0 = in_0.shape[1]
    C1 = in_1.shape[1]
    C2 = in_2.shape[1]
    C3 = in_3.shape[1]
    total_C = C0 + C1 + C2 + C3
    HW = in_0.shape[2] * in_0.shape[3]

    BLOCK_C = 64
    BLOCK_HW = 32

    # 2D grid: [4 inputs, max_channel_blocks_per_input]
    max_C = max(C0, C1, C2, C3)
    max_blocks = (max_C + BLOCK_C - 1) // BLOCK_C

    out = torch.empty((1, total_C), dtype=in_0.dtype, device=in_0.device)

    fused_2d_pool_kernel[(4, max_blocks)](
        in0_ptr=in_0, in1_ptr=in_1, in2_ptr=in_2, in3_ptr=in_3,
        out_ptr=out,
        C0=C0, C1=C1, C2=C2, C3=C3,
        HW=HW,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW,
    )

    return out


def replacement_func():
    return fused_cat_pool_flatten