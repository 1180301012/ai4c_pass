import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    # Match only the two arithmetic ops; downstream split/squeeze/contiguous
    # remain in the graph as cheap view ops.
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_scale_sub_kernel(
    in0_ptr,    # [1, N_ROWS, 1] int64 — stride (N_ROWS, 1, 1); data[i] = element [0,i,0]
    in1_ptr,    # [1, N_ROWS, 2] float16/bf16; data[i*2+k] = element [0,i,k]
    out_ptr,    # [1, N_ROWS, 2] float16/bf16; same layout as in1
    N_ROWS,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ROWS

    # in_0[0, i, 0]: stride-1 along dim-1 → flat offset = i
    in0_vals = tl.load(in0_ptr + offsets, mask=mask, other=0)
    # Scale to float
    scaled = in0_vals.to(tl.float32) * 1000000.0

    # in_1[0, i, 0] and in_1[0, i, 1]: flat offset = i*2 + k
    in1_0 = tl.load(in1_ptr + offsets * 2 + 0, mask=mask, other=0.0)
    in1_1 = tl.load(in1_ptr + offsets * 2 + 1, mask=mask, other=0.0)

    result_0 = in1_0.to(tl.float32) - scaled
    result_1 = in1_1.to(tl.float32) - scaled

    if IS_BF16:
        out0 = result_0.to(tl.bfloat16)
        out1 = result_1.to(tl.bfloat16)
    else:
        out0 = result_0.to(tl.float16)
        out1 = result_1.to(tl.float16)

    tl.store(out_ptr + offsets * 2 + 0, out0, mask=mask)
    tl.store(out_ptr + offsets * 2 + 1, out1, mask=mask)


@torch.fx.wrap
def triton_fused_mul_sub(in_0, in_1):
    # in_0: [1, N_ROWS, 1] int64  (may be on CPU)
    # in_1: [1, N_ROWS, 2] float16/bfloat16  (on CUDA)
    N_ROWS = in_1.shape[1]

    # Move in_0 to CUDA and cast to same dtype as in_1 (torch.as_tensor is allowed)
    in_0_cuda = torch.as_tensor(in_0, device=in_1.device, dtype=in_1.dtype)

    # Allocate output on CUDA with same shape/dtype as in_1
    out = torch.empty_like(in_1)

    BLOCK_SIZE = 32
    num_blocks = (N_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_scale_sub_kernel[(num_blocks,)](
        in_0_cuda,
        in_1,
        out,
        N_ROWS,
        IS_BF16=(in_1.dtype == torch.bfloat16),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return triton_fused_mul_sub