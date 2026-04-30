import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_3)


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def fused_relu_mean_kernel(
    input_ptr,
    relu_output_ptr,
    sum_output_ptr,
    NC,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    spatial_start = pid_hw * BLOCK_SIZE
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < HW

    global_offsets = pid_nc * HW + spatial_offsets

    # Load input
    x = tl.load(input_ptr + global_offsets, mask=spatial_mask, other=0.0)

    # ReLU
    relu_x = tl.maximum(x, 0.0)

    # Store ReLU output (cast to output dtype automatically)
    tl.store(relu_output_ptr + global_offsets, relu_x, mask=spatial_mask)

    # Accumulate partial sum in float32 for precision
    local_sum = tl.sum(relu_x.to(tl.float32))

    # Atomic add to channel sum buffer
    tl.atomic_add(sum_output_ptr + pid_nc, local_sum)


@triton.jit
def compute_mean_kernel(
    sum_ptr,
    mean_ptr,
    HW,
    NC,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NC

    # Load accumulated sum (float32)
    sum_val = tl.load(sum_ptr + offsets, mask=mask)

    # Compute mean by dividing by HW
    mean_val = sum_val / HW

    # Store mean (cast to output dtype automatically)
    tl.store(mean_ptr + offsets, mean_val, mask=mask)


@torch.fx.wrap
def fused_relu_mean_impl(input_tensor):
    shape = input_tensor.shape
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]
    NC = N * C
    HW = H * W

    # Allocate outputs
    relu_output = torch.empty_like(input_tensor)
    mean_output = torch.empty((N, C, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    sum_buffer = torch.zeros((NC,), dtype=torch.float32, device=input_tensor.device)

    # Fused ReLU + sum kernel
    BLOCK_SIZE = 256
    num_spatial_blocks = (HW + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_fused = (NC, num_spatial_blocks)

    fused_relu_mean_kernel[grid_fused](
        input_ptr=input_tensor,
        relu_output_ptr=relu_output,
        sum_output_ptr=sum_buffer,
        NC=NC,
        HW=HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Compute mean kernel
    MEAN_BLOCK_SIZE = 256
    num_mean_blocks = (NC + MEAN_BLOCK_SIZE - 1) // MEAN_BLOCK_SIZE

    compute_mean_kernel[(num_mean_blocks,)](
        sum_ptr=sum_buffer,
        mean_ptr=mean_output,
        HW=HW,
        NC=NC,
        BLOCK_SIZE=MEAN_BLOCK_SIZE,
    )

    return (relu_output, mean_output)


def replacement_func():
    return fused_relu_mean_impl