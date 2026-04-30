import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
    return tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_sigmoid_scale_relu_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid: one block per channel (512 blocks total)
    pid = tl.program_id(0)

    # Load sigmoid input for this channel - single scalar load
    sig_input = tl.load(in_0_ptr + pid)
    # Compute sigmoid in f32 for accuracy
    sig_val = tl.sigmoid(sig_input.to(tl.float32))
    scale = 1.0 + sig_val

    # Base offset for this channel
    base = pid * HW

    # Process HW elements in chunks of BLOCK_SIZE (unrolled loop)
    for i in tl.static_range(0, HW, BLOCK_SIZE):
        offsets = base + i + tl.arange(0, BLOCK_SIZE)
        # Load (coalesced, no mask needed)
        x = tl.load(in_1_ptr + offsets)
        # Compute: relu(x * scale)
        result = x.to(tl.float32) * scale
        result = tl.maximum(result, 0.0)
        # Store
        tl.store(out_ptr + offsets, result.to(x.dtype))


@torch.fx.wrap
def fused_sigmoid_scale_relu(in_0, in_1):
    # in_0: [1, 512], in_1: [1, 512, 64, 64]
    out = torch.empty_like(in_1)

    fused_sigmoid_scale_relu_kernel[(512,)](
        in_0,
        in_1,
        out,
        HW=4096,
        BLOCK_SIZE=512,
        num_warps=4,
        num_stages=8,
    )

    return out


def replacement_func():
    return fused_sigmoid_scale_relu