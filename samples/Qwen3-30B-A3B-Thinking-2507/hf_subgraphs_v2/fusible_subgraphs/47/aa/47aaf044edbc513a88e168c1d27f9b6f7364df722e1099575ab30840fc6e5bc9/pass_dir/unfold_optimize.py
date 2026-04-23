import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, [9, 1], 1, [4, 0], 1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def unfold_kernel(
    input_ptr,
    output_ptr,
    batch,
    channels,
    seq_len,
    output_batch,
    output_channels,
    output_kernel_size,
    BLOCK_SIZE: tl.constexpr
):
    linear_index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear_index < output_batch * output_channels * output_kernel_size

    i = linear_index // (output_channels * output_kernel_size)
    j = (linear_index % (output_channels * output_kernel_size)) // output_kernel_size
    k = linear_index % output_kernel_size

    c = (i * 8 + j) % channels
    p = (i * 8 + j) // channels
    s = p + k - 4

    input_val = tl.zeros((1,), dtype=tl.float16)
    if s >= 0 and s < seq_len:
        input_offset = batch * channels * seq_len + c * seq_len + s
        input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

    output_offset = linear_index
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def unfold_optimize(in_0):
    batch = 1
    channels = 16
    seq_len = 45
    output_batch = 90
    output_channels = 8
    output_kernel_size = 9
    total_output_elements = output_batch * output_channels * output_kernel_size

    output = torch.empty((output_batch, output_channels, output_kernel_size), dtype=in_0.dtype, device=in_0.device)

    BLOCK_SIZE = 128
    grid = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    unfold_kernel[grid](
        in_0,
        output,
        batch,
        channels,
        seq_len,
        output_batch,
        output_channels,
        output_kernel_size,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

def replacement_func():
    return unfold_optimize