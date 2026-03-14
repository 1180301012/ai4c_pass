import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the computation pattern:
    conv2d(in_3, in_1, in_0, stride, padding, dilation, groups) -> sigmoid -> view -> mul -> contiguous
    """
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract the arguments needed for the replacement function."""
    return (in_0, in_1, in_2, in_3)


# Single-pass fused kernel: compute conv + sigmoid + mul for each output element
@triton.jit
def fused_kernel(
    in_0_ptr,  # bias (96,)
    in_1_ptr,  # weight (96, 8, 1, 1)
    in_2_ptr,  # main input (1, 96, 128, 128)
    in_3_ptr,  # conv input (1, 32, 1, 1)
    out_ptr,
    N_elements,  # Total elements: 96 * 128 * 128 = 1572864
    N,           # Number of channels = 96
    H,           # Height = 128
    W,           # Width = 128
    stride_in_2_n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-pass fused kernel: conv + sigmoid + mul + contiguous.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_elements
    
    # Compute channel, height, width indices
    c = offs // (H * W)
    h = (offs % (H * W)) // W  
    w = offs % W
    
    # Compute grouped conv for this output channel
    group_id = c // 24
    in_ch_start = group_id * 8
    
    # Load bias for this channel
    bias = tl.load(in_0_ptr + c, mask=c < N, other=0.0)
    
    # Compute conv: sum over 8 input channels
    conv_out = bias
    for i in range(8):
        in_ch = in_ch_start + i
        weight_val = tl.load(in_1_ptr + c * 8 + i, mask=c < N, other=0.0)
        input_val = tl.load(in_3_ptr + in_ch, mask=c < N, other=0.0)
        conv_out = conv_out + weight_val * input_val
    
    # Apply sigmoid
    sigmoid_out = 1.0 / (1.0 + tl.exp(-conv_out))
    
    # Multiply with in_2
    base_offset = c * stride_in_2_n + h * W + w
    in_2_val = tl.load(in_2_ptr + base_offset, mask=mask, other=0.0)
    
    # Multiply and store
    result = in_2_val * sigmoid_out
    tl.store(out_ptr + base_offset, result, mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_mul_wrapper(in_0, in_1, in_2, in_3):
    """
    Full computation in Triton: conv2d + sigmoid + mul + contiguous.
    
    Args:
        in_0: bias tensor, shape (96,)
        in_1: weight tensor, shape (96, 8, 1, 1)
        in_2: main input tensor, shape (1, 96, 128, 128)
        in_3: conv input tensor, shape (1, 32, 1, 1)
    
    Returns:
        Output tensor, shape (1, 96, 128, 128)
    """
    N = in_2.shape[1]   # 96
    H = in_2.shape[2]   # 128
    W = in_2.shape[3]   # 128
    N_elements = N * H * W  # 1572864
    
    # Allocate output
    out = torch.empty_like(in_2)
    
    # Launch kernel - optimized block size
    BLOCK_SIZE = 512
    num_programs = (N_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_kernel[(num_programs,)](
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        N_elements,
        N,
        H,
        W,
        in_2.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    """Return the replacement function."""
    return fused_conv_sigmoid_mul_wrapper