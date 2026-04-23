import torch
import triton
import triton.language as tl


def pattern(conv_output):
    tmp_3 = conv_output.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(tmp_3.shape[0], -1, tmp_3.shape[3])
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return (tmp_5,)


def replacement_args(conv_output):
    return (conv_output,)


@triton.jit
def fused_permute_reshape_sigmoid_kernel(
    input_ptr,
    output_ptr,
    N,
    C,
    H,
    W,
    HW,
    CHW,
    HWC,
    total_elements,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel: reads NCHW, applies sigmoid, writes NHWC layout.
    Each program handles a 2D tile of (hw_range, c_range) for a given (n, c_block).
    Output writes are coalesced along the C dimension within each hw position.
    """
    pid_hw = tl.program_id(0)
    pid_n_c = tl.program_id(1)
    
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    pid_n = pid_n_c // num_c_blocks
    pid_c = pid_n_c % num_c_blocks
    
    hw_start = pid_hw * BLOCK_HW
    c_start = pid_c * BLOCK_C
    
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    
    mask = (hw_offsets[None, :] < HW) & (c_offsets[:, None] < C) & (pid_n < N)
    
    input_offsets = pid_n * CHW + c_offsets[:, None] * HW + hw_offsets[None, :]
    
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    result = tl.sigmoid(x)
    
    output_offsets = pid_n * HWC + hw_offsets[None, :] * C + c_offsets[:, None]
    
    tl.store(output_ptr + output_offsets, result, mask=mask)


@torch.fx.wrap
def _run_fused_kernel(conv2d_out):
    """Apply fused permute+reshape+sigmoid on conv2d output."""
    N, C, H, W = conv2d_out.shape
    HW = H * W
    CHW = C * H * W
    HWC = H * W * C
    
    out = torch.empty((N, HW, C), dtype=conv2d_out.dtype, device=conv2d_out.device)
    
    total_elements = N * C * H * W
    
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_HW = 128
    
    num_hw_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    num_n_c_blocks = N * num_c_blocks
    
    grid = (num_hw_blocks, num_n_c_blocks)
    
    fused_permute_reshape_sigmoid_kernel[grid](
        input_ptr=conv2d_out,
        output_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        HW=HW,
        CHW=CHW,
        HWC=HWC,
        total_elements=total_elements,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
    )
    
    return (out,)


def replacement_func():
    return _run_fused_kernel