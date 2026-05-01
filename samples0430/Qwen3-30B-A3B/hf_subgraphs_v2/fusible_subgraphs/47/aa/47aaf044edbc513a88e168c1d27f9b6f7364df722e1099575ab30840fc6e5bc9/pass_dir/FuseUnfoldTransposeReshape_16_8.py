import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def window_kernel(
    input_ptr,
    output_ptr,
    W: tl.int32,
    C: tl.int32,
    K: tl.int32,
    padding_left: tl.int32,
    kernel_size: tl.int32,
    stride: tl.int32,
    N: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.thread_id(0)
    idx = pid * BLOCK_SIZE + tid
    if idx >= N:
        return

    # Compute output indices (i, k, j)
    i = idx // (K * kernel_size)
    k = (idx % (K * kernel_size)) // kernel_size
    j = idx % kernel_size

    # Compute window position w and channel offset c
    c = i % (C // K)
    w = i // (C // K)

    # Input channel index
    input_channel = k * (C // K) + c

    # Position in input W dimension
    pos_in_w = w + padding_left + j

    # Load value with padding
    mask = (pos_in_w >= 0) & (pos_in_w < W)
    value = tl.load(
        input_ptr + input_channel * W + pos_in_w,
        mask=mask,
        other=0.0
    )

    # Store to output
    tl.store(output_ptr + idx, value)

@torch.fx.wrap
def window_op(in_0):
    batch, C, W = in_0.shape
    K = 8
    kernel_size = 9
    padding_left = 4
    padding_right = 0
    stride = 1
    L = (W + padding_left + padding_right - kernel_size) // stride + 1
    N = L * C * kernel_size
    output = torch.empty((L * (C // K), K, kernel_size), dtype=in_0.dtype, device=in_0.device)
    output_flat = output.view(-1)
    BLOCK_SIZE = 128
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    window_kernel[(num_blocks,)](
        in_0,
        output_flat,
        W,
        C,
        K,
        padding_left,
        kernel_size,
        stride,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def replacement_func():
    return window_op