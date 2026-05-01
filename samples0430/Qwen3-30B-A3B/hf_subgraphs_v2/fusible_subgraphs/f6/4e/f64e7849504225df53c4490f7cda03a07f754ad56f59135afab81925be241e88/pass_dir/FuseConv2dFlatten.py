import torch
import triton
import triton.language as tl


def pattern(in_2 : torch.Tensor, in_1 : torch.Tensor, in_0 : torch.Tensor):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return torch.flatten(conv2d, 2)

def replacement_args(in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
    return (in_0, in_1, in_2)


@triton.jit
def conv2d_flatten_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, C_in, C_out, H, W,
    BLOCK_COUT: tl.constexpr,
    BLOCK_HW: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_hw = tl.program_id(2)

    start_cout = pid_cout * BLOCK_COUT
    start_hw = pid_hw * BLOCK_HW

    Y_block = Y_ptr + pid_n * (C_out * H * W) + start_cout * H * W + start_hw

    bias = tl.load(B_ptr + start_cout, mask=tl.arange(0, BLOCK_COUT) < C_out - start_cout)

    for h_offset in range(BLOCK_HW):
        h = (start_hw + h_offset) // W
        w = (start_hw + h_offset) % W

        X_block = X_ptr + pid_n * C_in * H * W + h * W * C_in + w * C_in
        acc = tl.zeros((BLOCK_COUT,), dtype=tl.float32)

        for cin in range(C_in):
            x = tl.load(X_block + cin, mask=cin < C_in)
            w = tl.load(W_ptr + start_cout * C_in + cin, mask=tl.arange(0, BLOCK_COUT) < C_out - start_cout)
            acc += x * w

        acc += bias
        tl.store(Y_block + h_offset, acc, mask=tl.arange(0, BLOCK_COUT) < C_out - start_cout)


@torch.fx.wrap
def conv2d_flatten(in_0, in_1, in_2):
    batch, in_ch, H, W = in_2.shape
    out_ch = in_1.shape[0]

    output = torch.empty((batch, out_ch * H * W), dtype=in_2.dtype, device=in_2.device)

    BLOCK_COUT = 32
    BLOCK_HW = 64

    grid_n = batch
    grid_cout = (out_ch + BLOCK_COUT - 1) // BLOCK_COUT
    grid_hw = (H * W + BLOCK_HW - 1) // BLOCK_HW

    conv2d_flatten_kernel[(grid_n, grid_cout, grid_hw)](
        in_2, in_1, in_0, output,
        batch, in_ch, out_ch, H, W,
        BLOCK_COUT=BLOCK_COUT, BLOCK_HW=BLOCK_HW
    )
    return output

def replacement_func():
    return conv2d_flatten